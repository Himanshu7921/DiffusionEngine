import random
from pathlib import Path

import numpy as np
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
        epochs=100,
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
        use_amp=True
    ):

        self.ddpm = ddpm

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.config = config

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

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=15,
            threshold=1e-3,
            threshold_mode="rel",
            min_lr=1e-6
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
        print(f"Dataset size:    {len(dataset)}")
        print(f"Training images: {train_size}")
        print(f"Validation:      {val_size}")
        print(f"Batch size:      {batch_size}")
        print(f"Image size:      {image_size}x{image_size}")
        print(f"Training steps:  {len(self.train_loader)} per epoch")
        print(f"W&B run:         {self.run.url}")
        print("=" * 60)


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

                pred_noise, actual_noise = self.ddpm(x_0)

                loss = F.mse_loss(
                    pred_noise.float(),
                    actual_noise.float()
                )

            # Backpropagation.
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ddpm.parameters(),
                max_norm=1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

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

                wandb.log({
                    "global_step": self.global_step,
                    "train/batch_loss": loss_value,
                    "train/gradient_norm": float(grad_norm),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"]
                })

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

        print("Fixed validation noise and timesteps generated.")


    @torch.no_grad()
    def validate(self, epoch):

        self.ddpm.eval()

        if self.vae is not None:
            self.vae.eval()

        running_loss = 0.0

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

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):

                pred_noise, actual_noise = self.ddpm(
                    x_0,
                    timesteps=timesteps,
                    noise=noise
                )

                loss = F.mse_loss(
                    pred_noise.float(),
                    actual_noise.float()
                )

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
            self.lr_scheduler.step(val_loss)

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

