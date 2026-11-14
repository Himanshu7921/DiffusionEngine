import random
from pathlib import Path
import math
import numpy as np
import copy # for maintaining EMA of DDPM's params
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_loader import CelebADataset, LatentDataset

from diffusers import AutoencoderKL

class Trainer:
    def __init__(
        self,
        ddpm,
        config,
        dataset_path="data/celeba/img_align_celeba",
        batch_size=32,
        image_size=64,
        epochs = 100,
        warmup_epochs = 10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        validation_split=0.05,
        num_workers=4,
        checkpoint_dir="./checkpoint",
        checkpoint_every=5,
        log_every=50,
        project_name="DiffusionLab",
        run_name="DDPM-CelebA",
        seed=42,
        use_min_snr_loss: bool = True,
        use_amp=True
    ):

        self.ddpm = ddpm

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


        self.config = config
        self.use_min_snr_loss = use_min_snr_loss # paper: https://arxiv.org/pdf/2303.09556 | Solves unstable DDPM's training

        self.latent_denoise = config["latent_denoise"]
        self.vae = None
        self.scaling_factor = None
        self.latent_denoise = config["latent_denoise"]
        self.precomputed_latents = config.get("precomputed_latents", False)

        if self.precomputed_latents and not self.latent_denoise:
            raise ValueError(
                "precomputed_latents=True requires latent_denoise=True"
            )

        self.vae = None
        self.scaling_factor = None

        # Load VAE only when encoding is required during training.
        if self.latent_denoise and not self.precomputed_latents:
            self.vae = AutoencoderKL.from_pretrained(
                config["vae_model"]
            ).to(self.device)

            self.vae.eval()
            self.vae.requires_grad_(False)

            self.scaling_factor = self.vae.config.scaling_factor

        self.batch_size = batch_size
        self.image_size = image_size
        self.epochs = epochs

        self.learning_rate = learning_rate
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


        self.use_amp = use_amp and self.device.type == "cuda"

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.train_losses = []
        self.val_losses = []

        self._set_seed(seed)

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------
        self.ddpm = self.ddpm.to(self.device)
        self.injector = self.ddpm.noise_injection_pipieline.noise_injector
        self.alpha = self.injector.alpha.to(self.device)
        self.alpha_bar = self.injector.alpha_bar.to(self.device)
        self.beta = self.injector.beta.to(self.device)

        # for maintaining a EMA of DDPM's param
        self.ema_ddpm = copy.deepcopy(self.ddpm)
        self.ema_decay = 0.9999
        self.ema_ddpm.eval()
        self.ema_ddpm.requires_grad_(False)


        # ----------------------------------------------------
        # Dataset and DataLoaders
        # ----------------------------------------------------

        if self.latent_denoise and self.precomputed_latents:
            dataset = LatentDataset(
                latent_path=dataset_path
            )
        else:
            dataset = CelebADataset(
                root_dir=dataset_path,
                image_size=image_size
            )

        val_size = max(
            1,
            int(len(dataset) * validation_split)
        )

        train_size = len(dataset) - val_size

        generator = torch.Generator().manual_seed(seed)

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
            "drop_last": True
        }

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            **loader_kwargs
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory = self.device.type == "cuda",
            drop_last=False,
            persistent_workers=num_workers > 0
        )
        self.create_fixed_validation_inputs(seed = 12345)

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.ddpm.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = epochs - warmup_epochs,
            eta_min=1e-5
        )

        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler
            ],
            milestones=[warmup_epochs]
        )

        # ----------------------------------------------------
        # Mixed Precision
        # ----------------------------------------------------

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp
        )

        # ----------------------------------------------------
        # W&B
        # ----------------------------------------------------

        self.run = wandb.init(
            project=project_name,
            name=run_name,
            # id="wjh3gscw",
            config={
                **config,
                "batch_size": batch_size,
                "image_size": image_size,
                "epochs": epochs,
                "latent_denoise": self.latent_denoise,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer": "AdamW",
                "loss": "MSE",
                "precomputed_latents": self.precomputed_latents,
                "dataset": "CelebA",
                "validation_split": validation_split,
                "mixed_precision": self.use_amp
            },
            resume="allow"
        )

        self.wandb_run_id = self.run.id

        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("validation/*", step_metric="epoch")

        print("=" * 60)
        print("DDPM TRAINER INITIALIZED")
        print("=" * 60)
        print(f"Device:          {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Latent Denoise : {'Enabled' if self.latent_denoise else 'Disabled'}")
        print(f"Precomputed Latents: {'Enabled' if self.precomputed_latents else 'Disabled'}")
        print(f"Loss: {'MIN-SNR-Loss' if self.use_min_snr_loss else 'MSE-Loss'}")
        print(f"Scheduler: {self.config['scheduler']}")
        print(f"Dataset size:    {len(dataset)}")
        print(f"Training images: {train_size}")
        print(f"Validation:      {val_size}")
        print(f"Batch size:      {batch_size}")
        print(f"Image size:      {image_size}x{image_size}")
        print(f"Training steps:  {len(self.train_loader)} per epoch")
        print(f"W&B run:         {self.run.url}")
        print("=" * 60)

    def log_training_metrics(
        self,
        *,
        pred_noise,
        fraction,
        timesteps,
        raw_noise_mse,
        simple_loss,
        vlb_loss,
        hybrid_loss,
        grad_norm,
    ):
        """Log diffusion training diagnostics to Weights & Biases."""

        lambda_vlb = self.config.get("lambda_vlb", 0.001)
        weighted_vlb = lambda_vlb * vlb_loss

        with torch.no_grad():
            if fraction is not None:
                log_var = self.ddpm_trainer_log_var(fraction, timesteps)

            metrics = {
                # Loss components
                "train/loss/raw_noise_mse": raw_noise_mse.detach().item(),
                "train/loss/simple": simple_loss.detach().item(),
                "train/loss/vlb": vlb_loss.detach().item(),
                "train/loss/weighted_vlb": weighted_vlb.detach().item(),
                "train/loss/hybrid": hybrid_loss.detach().item(),

                # Loss contribution
                "train/loss/vlb_to_simple_ratio": (
                    weighted_vlb.detach().abs()
                    / simple_loss.detach().abs().clamp(min=1e-12)
                ).item(),

                # Gradient diagnostics
                "train/gradients/noise_output_norm": (
                    pred_noise.grad.detach().float().norm().item()
                    if pred_noise.grad is not None else 0.0
                ),
                "train/gradients/variance_output_norm": (
                    fraction.grad.detach().float().norm().item()
                    if fraction.grad is not None else 0.0
                ),
                "train/gradients/model_norm_before_clipping": float(grad_norm),

                # Learned variance diagnostics
                "train/variance/log_var_mean": log_var.mean().item(),
                "train/variance/log_var_min": log_var.min().item(),
                "train/variance/log_var_max": log_var.max().item(),
                "train/variance/predicted_var_mean": log_var.exp().mean().item(),
                "train/variance/fraction_mean": fraction.detach().float().mean().item(),
                "train/variance/fraction_min": fraction.detach().float().min().item(),
                "train/variance/fraction_max": fraction.detach().float().max().item(),

                # Timestep diagnostics
                "train/timesteps/mean": timesteps.float().mean().item(),
                "train/timesteps/min": timesteps.min().item(),
                "train/timesteps/max": timesteps.max().item(),
                "train/timesteps/t1_fraction": (
                    (timesteps == 1).float().mean().item()
                ),

                # Optimization
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/amp/scale": self.scaler.get_scale(),
            }

            wandb.log({
                "global_step": self.global_step,
                **metrics,
            })

    def ddpm_trainer_log_var(self, fraction, timesteps):
        """Return predicted log variance for monitoring."""
        return self._Trainer__get_predicted_log_var(
            fraction.detach().float(),
            timesteps,
        ).detach().float()


    def validate_diffusion_losses(self):
        """
        works only for Improved DDPM
        """
        self.ddpm.train()

        batch = next(iter(self.train_loader))
        x_0 = self.prepare_input(batch, training=True).to(
            self.device
        )

        # Forward pass without AMP
        with torch.autocast(
            device_type=self.device.type,
            enabled=False
        ):
            timesteps = torch.tensor(
                [1, 2, 100, 799],
                device=self.device,
                dtype=torch.long
            )
            pred_noise, actual_noise, x_t, timesteps, fraction = self.ddpm(x_0.float(), timesteps = timesteps)
            log_var = self.__get_predicted_log_var(fraction, timesteps)
            print("Log variance min:", log_var.min().item())
            print("Log variance max:", log_var.max().item())
            print("Log variance mean:", log_var.mean().item())

            pred_noise = pred_noise.float()
            actual_noise = actual_noise.float()
            fraction = fraction.float()
            x_t = x_t.float()
            x_0 = x_0.float()

            # Simple loss
            L_simple = F.mse_loss(pred_noise, actual_noise)

            # VLB loss
            _, L_vlb, L_hybrid = self.__get_hybrid_loss(
                x_0=x_0,
                x_t=x_t,
                pred_noise=pred_noise,
                pred_variance=fraction,
                actual_noise=actual_noise,
                timesteps=timesteps
            )

            lambda_vlb = self.config.get("lambda_vlb", 0.001)

            weighted_vlb = lambda_vlb * L_vlb

        # Check numerical stability
        losses = {
            "L_simple": L_simple,
            "L_vlb": L_vlb,
            "weighted_vlb": weighted_vlb,
            "L_hybrid": L_hybrid
        }

        print("\n--- LOSS MAGNITUDES ---")

        for name, value in losses.items():
            print(
                f"{name:15s}: {value.detach().item():.8f} "
                f"| finite: {torch.isfinite(value).all().item()}"
            )

        # Check relative contribution
        print(
            "\nVLB / Simple:",
            weighted_vlb.detach().item()
            / (abs(L_simple.detach().item()) + 1e-12)
        )

        # Check gradients independently
        grad_simple = torch.autograd.grad(
            L_simple,
            (pred_noise, fraction),
            retain_graph=True,
            allow_unused=True
        )

        grad_vlb = torch.autograd.grad(
            L_vlb,
            (pred_noise, fraction),
            retain_graph=True,
            allow_unused=True
        )

        def grad_norm(gradient):
            if gradient is None:
                return 0.0

            return gradient.detach().float().norm().item()

        print("\n--- GRADIENT VALIDATION ---")

        print(
            "L_simple -> noise:",
            grad_norm(grad_simple[0])
        )

        print(
            "L_simple -> variance:",
            grad_norm(grad_simple[1])
        )

        print(
            "L_vlb -> noise:",
            grad_norm(grad_vlb[0])
        )

        print(
            "L_vlb -> variance:",
            grad_norm(grad_vlb[1])
        )

        # Check timestep distribution
        print("\n--- TIMESTEP DISTRIBUTION ---")
        print("Minimum timestep:", timesteps.min().item())
        print("Maximum timestep:", timesteps.max().item())
        print("t = 1 samples:", (timesteps == 1).sum().item())
        print("t > 1 samples:", (timesteps > 1).sum().item())

        return losses


    def __get_hybrid_loss(self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        pred_noise: torch.Tensor,
        pred_variance: torch.Tensor,
        actual_noise: torch.Tensor,
        timesteps: torch.Tensor,
        simple_loss: torch.Tensor = None
    ):
        """
        Computes the hybrid loss proposed in Improved Denoising Diffusion
        Probabilistic Models (Nichol & Dhariwal, 2021).
        > L_hybrid = L_simple + lambda * L_vlb

        Combines the simple noise prediction loss and variational lower
        bound (VLB) loss to train the model's mean and learned variance.

        Notes:
            - L_simple trains the reverse process mean.
            - L_vlb trains the learned variance without propagating
            gradients through the mean prediction.
        
        > pred_noise: (B, C, H, W)
        > actual_noise: (B, C, H, W)
        > timesteps: (B, )

        """
        assert torch.all(timesteps >= 1)
        assert torch.all(timesteps <= self.beta.shape[0] - 1)

        pred_noise = pred_noise.float()
        actual_noise = actual_noise.float()

        x_0 = x_0.float()
        x_t = x_t.float()

        if simple_loss is None:
            simple_loss = F.mse_loss(pred_noise, actual_noise)

        pred_noise = pred_noise.float()
        pred_variance = pred_variance.float()
        actual_noise = actual_noise.float()

        # Simple loss

        # Timesteps masks
        L_0_mask = timesteps == 1
        L_t_mask = timesteps > 1

        L_vlb = torch.zeros(
            x_0.shape[0], device = x_0.device, dtype = x_0.dtype
        )

        if L_0_mask.any():
            L_vlb[L_0_mask] = self.__get_decoder_nll(
                x_0 = x_0[L_0_mask],
                x_t = x_t[L_0_mask],
                predicted_noise = pred_noise[L_0_mask],
                predicted_var = pred_variance[L_0_mask],
                timesteps = timesteps[L_0_mask]
            )

        if L_t_mask.any():
            L_vlb[L_t_mask] = self.__get_posterior_kl(
                x_0 = x_0[L_t_mask],
                x_t = x_t[L_t_mask],
                predicted_noise = pred_noise[L_t_mask],
                predicted_var = pred_variance[L_t_mask],
                timesteps = timesteps[L_t_mask]
            )
        

        L_vlb = L_vlb.mean()
        lambda_vlb = self.config.get("lambda_vlb", 0.001)
        L_hybrid = simple_loss + lambda_vlb * L_vlb
        return simple_loss, L_vlb, L_hybrid


    def __extract(self, value: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor):
        """
        extracts the specific sample for specific timesteps, requires because we need to calcuate L_vlb
        and reshapes for broadcasting

        sample: (B, C, H, W) (image or latent tensor)
        timestpes: (B, )
        """
        return value[timesteps].to(device = x.device, dtype = x.dtype).view(-1, 1, 1, 1)


    def __get_posterior_var(self):
        """
        computes posterior variance and clipped posterior variance for log interpolation

        """
        # calculate the posterior's variance for all timesteps
        beta_tilde = (self.beta[1:] * (1.0 - self.alpha_bar[:-1]) / (1.0 - self.alpha_bar[1:])) # Shape: (T, )

        posterior_variance = torch.cat([ # index-0 is a place holder to preserve (T+1, ) shape 
            torch.zeros_like(beta_tilde[:1]),
            beta_tilde
        ])

        # Replace t=1 variance with t=2 variance for log stability
        posterior_variance_clipped = torch.cat([
            torch.zeros_like(beta_tilde[:1]),
            beta_tilde[1:2],
            beta_tilde[1:]
        ])

        return posterior_variance, posterior_variance_clipped
    
    def __get_predicted_log_var(self,
        pred_variance: torch.Tensor,
        timesteps: torch.Tensor
    ):
        """
        computes predicted log_var mentioned in the Imporved DDPM's paper, by interpolating in the log-domain
        """
        _, posterior_variance_clipped = self.__get_posterior_var()
        beta_t = self.__extract(self.beta, timesteps, pred_variance)
        beta_tilde_t = self.__extract(posterior_variance_clipped, timesteps, pred_variance)
        log_beta_t = torch.log(beta_t)
        log_beta_tilde_t = torch.log(beta_tilde_t)
        log_var = pred_variance * log_beta_t + (1.0 - pred_variance) * log_beta_tilde_t # log domain interpolation
        return log_var

    def __get_decoder_nll(self,
                x_0: torch.Tensor,
                x_t: torch.Tensor,
                predicted_noise: torch.Tensor,
                predicted_var: torch.Tensor,
                timesteps: torch.Tensor,
        ):
        """
        computes L_0, mentioned in: Improved Denoising Diffusion
        Probabilistic Models (Nichol & Dhariwal, 2021).
        """

        # Diagnostic statistics
        log_2_pi = math.log(2.0 * math.pi)
        log_var = self.__get_predicted_log_var(predicted_var, timesteps = timesteps)

        alpha_t = self.__extract(self.alpha, timesteps, predicted_var) # (T+1, )
        alpha_bar_t = self.__extract(self.alpha_bar, timesteps, predicted_var) # (T+1, )
        beta_t = self.__extract(self.beta, timesteps, predicted_var) # (T+1, )

        reverse_mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t * predicted_noise / (torch.sqrt(1.0 - alpha_bar_t)))
        reverse_mean = reverse_mean.detach()
        L_0 = 1/2 * (log_2_pi + log_var + (x_0 - reverse_mean) ** 2 / torch.exp(log_var))

        # residual = x_0 - reverse_mean
        # variance = torch.exp(log_var)

        # print("\n--- DECODER NLL DIAGNOSTICS ---")

        # print("Residual RMS:", residual.detach().square().mean().sqrt().item())

        # print("Mean absolute residual:",
        #     residual.detach().abs().mean().item())

        # print("Predicted variance mean:",
        #     variance.detach().mean().item())

        # print("Predicted variance min:",
        #     variance.detach().min().item())

        # print("Predicted variance max:",
        #     variance.detach().max().item())

        # print("Log variance mean:",
        #     log_var.detach().mean().item())

        # print("NLL per element:",
        #     (L_0.detach().flatten(1).sum(dim=1)
        #     / L_0[0].numel()).mean().item())

        # log_term = 0.5 * (
        #     math.log(2.0 * math.pi) + log_var
        # )

        # squared_error_term = 0.5 * (
        #     (x_0 - reverse_mean).square()
        #     / torch.exp(log_var)
        # )

        # print("Log term mean:", log_term.detach().mean().item())
        # print("Squared error term mean:",
        #     squared_error_term.detach().mean().item())
        
        return L_0.flatten(1).sum(dim = 1)

    def __get_posterior_kl(self,
            x_0: torch.Tensor,
            x_t: torch.Tensor,
            predicted_noise: torch.Tensor,
            predicted_var: torch.Tensor,
            timesteps: torch.Tensor
        ):
        """
        computes L_{t-1}, mentioned in: Improved Denoising Diffusion
        Probabilistic Models (Nichol & Dhariwal, 2021).
        """

        alpha_t = self.__extract(self.alpha, timesteps, x_t)
        alpha_bar_t = self.__extract(self.alpha_bar, timesteps, x_t)
        beta_t = self.__extract(self.beta, timesteps, x_t)
        alpha_bar_prev = self.__extract(self.alpha_bar, timesteps - 1, x_t)

        # calculate reverse mean (detach reverse mean's grad) and var
        reverse_mean = (x_t - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
        reverse_mean = reverse_mean.detach() # Stop gradients through the predicted mean (reverse_mean) for VLB

        # Learned reverse variance
        log_pred_var = self.__get_predicted_log_var(predicted_var, timesteps)

        # calculate posterior mean
        coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        posterior_mean = coef1 * x_0 + coef2 * x_t


        # calculate posterior variance (beta_tilde)
        posterior_var, _ = self.__get_posterior_var()
        posterior_var = self.__extract(posterior_var, timesteps, x_t)
        log_posterior_var = torch.log(posterior_var)

        L_t_1 = (1/2) * (log_pred_var - log_posterior_var + (posterior_var + (posterior_mean  - reverse_mean) ** 2) / torch.exp(log_pred_var) - 1.0)
        return L_t_1.flatten(1).sum(dim = 1)


    def min_snr_loss(self, pred_noise: torch.Tensor, actual_noise: torch.Tensor, timesteps: torch.Tensor, alpha_cumprod: torch.Tensor, gamma: float = 5.0):
        """
        re-weights the contributions of loss accross different timesteps, according to the method discussed in the below paper
        > Efficient Diffusion Training via Min-SNR Weighting Strategy: https://arxiv.org/pdf/2303.09556
        """
        # Ensure numerically stable calculations
        pred_noise = pred_noise.float()
        actual_noise = actual_noise.float()
        alpha_bar = alpha_cumprod[timesteps].float()

        # Calculate Signal-to-Noise Ratio (SNR)
        snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)

        # re-calculate weights according to SNR
        weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp(1e-8)

        loss = F.mse_loss(
            pred_noise, actual_noise, reduction = 'none'
        ).mean(dim = (1, 2, 3))

        return (loss * weights).mean()

    @torch.no_grad()
    def update_ema(self):
        for ema_param, model_param in zip(self.ema_ddpm.parameters(), self.ddpm.parameters()):
            ema_param.lerp_(model_param, 1.0 - self.ema_decay)


    @torch.no_grad()
    def prepare_input(self, images, training=True):
        images = images.to(
            self.device,
            non_blocking=True
        )

        # Already encoded and scaled latent representations.
        if self.precomputed_latents:
            return images

        # Pixel-space diffusion.
        if not self.latent_denoise:
            return images

        # Encode RGB images using the pretrained VAE.
        posterior = self.vae.encode(images).latent_dist

        latents = (
            posterior.sample()
            if training
            else posterior.mode()
        )

        return latents * self.scaling_factor

    # ========================================================
    # Reproducibility
    # ========================================================

    def _set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ========================================================
    # Training
    # ========================================================

    def train_one_epoch(self, epoch):

        self.ddpm.train()

        running_loss = 0.0
        running_grad_norm = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs}"
        )

        for batch_idx, x_0 in enumerate(progress):
            x_0 = self.prepare_input(x_0, training=True) # if self.latent_denoise is enabled then this will be z_0

            x_0 = x_0.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision.
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):

                pred_noise, actual_noise, x_t, timesteps, fraction = self.ddpm(x_0)

                # Retain gradients on model outputs for diagnostics.
                pred_noise.retain_grad()
                if fraction is not None:
                    fraction.retain_grad()

                # Calculate raw noise prediction MSE independently.
                raw_noise_mse = F.mse_loss( # only for wandb loggin and debugging
                    pred_noise.float(),
                    actual_noise.float(),
                )

                if not self.use_min_snr_loss:
                    simple_loss = F.mse_loss(pred_noise.float(),actual_noise.float())
                else:
                    alpha_cumprod = self.ddpm.noise_injection_pipieline.noise_injector.alpha_bar
                    simple_loss = self.min_snr_loss(
                        pred_noise=pred_noise,
                        actual_noise=actual_noise,
                        timesteps = timesteps,
                        alpha_cumprod = alpha_cumprod,
                        gamma=5.0
                    )

                if self.config["sampler"].lower() == "improved_ddpm":
                    # Calculate the hybrid objective
                    L_simple, L_vlb, loss = self.__get_hybrid_loss(
                        x_0=x_0,
                        x_t=x_t,
                        pred_noise=pred_noise,
                        pred_variance=fraction,
                        actual_noise=actual_noise,
                        timesteps=timesteps,
                        simple_loss = simple_loss
                    )
                else:
                    L_simple = simple_loss
                    L_vlb = torch.zeros_like(simple_loss)
                    loss = simple_loss


            # Backpropagation.
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ddpm.parameters(),
                max_norm=1.0
            )

            old_scale = self.scaler.get_scale() # for EMA

            self.scaler.step(self.optimizer)
            self.scaler.update()

            new_scale = self.scaler.get_scale() # for EMA

            # Update EMA only if optimizer step was not skipped.
            if new_scale >= old_scale:
                self.update_ema()

            loss_value = loss.item()

            running_loss += loss_value
            running_grad_norm += float(grad_norm)

            self.global_step += 1

            progress.set_postfix(
                loss=f"{loss_value:.6f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Batch-level W&B logging.
            if self.global_step % self.log_every == 0:
                self.log_training_metrics(
                pred_noise=pred_noise,
                fraction=fraction,
                timesteps=timesteps,
                raw_noise_mse=raw_noise_mse,
                simple_loss=L_simple,
                vlb_loss=L_vlb,
                hybrid_loss=loss,
                grad_norm=grad_norm,
            )

        epoch_loss = running_loss / len(self.train_loader)

        avg_grad_norm = running_grad_norm / len(self.train_loader)

        return epoch_loss, avg_grad_norm

    # ========================================================
    # Validation
    # ========================================================

    def create_fixed_validation_inputs(self, seed = 12345):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        self.fixed_val_noise = []
        self.fixed_val_timesteps = []

        T = self.config["T"]

        for images in self.val_loader:
            batch_size = images.shape[0]

            noise = torch.randn(
                images.shape,
                generator=generator
            )

            timesteps = torch.randint(
                low=1,
                high=T + 1,
                size=(batch_size,),
                generator=generator
            )

            self.fixed_val_noise.append(noise)
            self.fixed_val_timesteps.append(timesteps)

    @torch.no_grad()
    def validate(self, epoch):

        self.ddpm.eval()

        if self.vae is not None:
            self.vae.eval()

        running_loss = 0.0
        alpha_cumprod = self.ddpm.noise_injection_pipieline.noise_injector.alpha_bar

        for batch_idx, x_0 in enumerate(
            tqdm(
                self.val_loader,
                desc="Validation",
                leave=False
            )
        ):

            x_0 = self.prepare_input(
                x_0,
                training=False
            )

            # Retrieve fixed validation inputs.
            noise = self.fixed_val_noise[batch_idx].to(
                self.device,
                non_blocking=True
            )

            timesteps = self.fixed_val_timesteps[batch_idx].to(
                self.device,
                non_blocking=True
            )

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                pred_noise, actual_noise, x_t, _ , fraction = self.ddpm(
                    x_0,
                    timesteps=timesteps,
                    noise=noise
                )

                raw_noise_mse = F.mse_loss(
                    pred_noise.float(),
                    actual_noise.float(),
                )

                if self.config["use_min_snr_loss"]:
                    simple_loss = self.min_snr_loss(
                        pred_noise=pred_noise,
                        actual_noise=actual_noise,
                        timesteps=timesteps,
                        alpha_cumprod=alpha_cumprod,
                        gamma=5.0
                    )
                else:
                    simple_loss = raw_noise_mse

                if self.config["sampler"].lower() == "improved_ddpm":
                    # Calculate the hybrid objective
                    L_simple, L_vlb, loss = self.__get_hybrid_loss(
                        x_0=x_0,
                        x_t=x_t,
                        pred_noise=pred_noise,
                        pred_variance=fraction,
                        actual_noise=actual_noise,
                        timesteps=timesteps,
                        simple_loss = simple_loss
                    )
                else:
                    L_simple = simple_loss
                    L_vlb = torch.zeros_like(simple_loss)
                    loss = simple_loss

            running_loss += loss.item()

        return running_loss / len(self.val_loader)

    # ========================================================
    # Checkpointing
    # ========================================================

    def save_checkpoint(self, filename):
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.ddpm.state_dict(),
            "ema_model_state_dict": self.ema_ddpm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "wandb_run_id": self.wandb_run_id,
            "config": self.config
        }

        save_path = self.checkpoint_dir / filename

        torch.save(checkpoint, save_path)

        print(f"Checkpoint saved: {save_path}")

    # ========================================================
    # Resume Training
    # ========================================================

    def resume(self, checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # restore the ema_ddpm's param
        self.ema_ddpm.load_state_dict(
            checkpoint["ema_model_state_dict"]
        )

        self.ddpm.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.lr_scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        self.scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        self.start_epoch = checkpoint["epoch"]

        self.global_step = checkpoint["global_step"]

        self.best_val_loss = checkpoint["best_val_loss"]

        self.train_losses = checkpoint["train_losses"]

        self.val_losses = checkpoint["val_losses"]

        print("=" * 60)
        print("TRAINING RESUMED")
        print("=" * 60)
        print(f"Starting epoch: {self.start_epoch + 1}")
        print(f"Global step:    {self.global_step}")
        print(f"Best val loss:  {self.best_val_loss:.6f}")
        print("=" * 60)

    # ========================================================
    # Loss Curve Logging
    # ========================================================

    def log_loss_curves(self, epoch):

        fig, ax = plt.subplots(figsize=(9, 5))

        epochs = range(1, len(self.train_losses) + 1)

        ax.plot(
            epochs,
            self.train_losses,
            label="Training Loss"
        )

        ax.plot(
            epochs,
            self.val_losses,
            label="Validation Loss"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("DDPM Training and Validation Loss")

        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        wandb.log({
            "epoch": epoch + 1,
            "loss_curves": wandb.Image(fig)
        })

        plt.close(fig)

    # ========================================================
    # Complete Training Loop
    # ========================================================

    def fit(self, resume_from=None):

        # Resume model and optimizer state if requested.
        if resume_from is not None:

            self.resume(resume_from)

        for epoch in range(self.start_epoch, self.epochs):

            self.current_epoch = epoch

            # Training.
            train_loss, grad_norm = self.train_one_epoch(epoch)

            # Validation.
            val_loss = self.validate(epoch)

            # Update scheduler using validation loss.
            self.lr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print("\n" + "=" * 60)
            print(f"Epoch:       {epoch + 1}/{self.epochs}")
            print(f"Train Loss:  {train_loss:.6f}")
            print(f"Val Loss:    {val_loss:.6f}")
            print(f"Grad Norm:   {grad_norm:.6f}")
            print(f"LR:          {current_lr:.2e}")
            print("=" * 60)

            # Epoch-level W&B logging.
            wandb.log({
                "epoch": epoch + 1,
                "global_step": self.global_step,

                "train/epoch_loss": train_loss,
                "validation/epoch_loss": val_loss,

                "train/epoch_gradient_norm": grad_norm,
                "train/learning_rate": current_lr
            })

            # Save best model based on validation loss.
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                self.save_checkpoint("best.pt")

                print(
                    f"New best validation loss: "
                    f"{self.best_val_loss:.6f}"
                )

            # Save latest checkpoint every epoch.
            self.save_checkpoint("latest.pt")

            # Periodic checkpoints.
            if (epoch + 1) % self.checkpoint_every == 0:

                self.save_checkpoint(
                    f"epoch_{epoch + 1}.pt"
                )

            # Log complete loss curves.
            self.log_loss_curves(epoch)

        print("\nTraining completed.")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        wandb.finish()

