import torch
import argparse
from config import config
from diffusers import AutoencoderKL


@torch.no_grad()
def decode_output(generated, config, vae=None):
    if config["latent_denoise"]:
        if vae is None:
            raise ValueError(
                "VAE must be provided when latent_denoise=True"
            )

        generated = generated / vae.config.scaling_factor
        generated = vae.decode(generated).sample

    return ((generated + 1) / 2).clamp(0, 1)

@torch.no_grad()
def sample(ddpm, config, device, batch_size, vae = None):
    ddpm.eval()

    T = config["T"]

    latent_denoise = config["latent_denoise"]

    # Determine initial noise dimensions.
    image_size = config["image_size"]

    if latent_denoise:
        if vae is None:
            raise ValueError(
                "VAE must be provided when latent_denoise=True"
            )

        downsample_factor = 2 ** (
            len(vae.config.block_out_channels) - 1
        )

        if image_size % downsample_factor != 0:
            raise ValueError("Image size must be divisible by VAE downsample factor")

        height = image_size // downsample_factor
        width = image_size // downsample_factor


    else:
        height = image_size
        width = image_size

    channels = config["in_channels"]

    # Start from Gaussian noise with the same shape as the training image
    x_t = torch.randn(batch_size, channels, height, width).to(device)

    injector = ddpm.noise_injection_pipieline.noise_injector

    # Move schedules to the sampling device.
    beta = injector.beta.to(device)
    alpha = injector.alpha.to(device)
    alpha_bar = injector.alpha_bar.to(device)

    for t in range(T, 0, -1):

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise.
        eps_theta = ddpm.noise_prediction_pipeline(x_t, t_tensor)

        # Extract schedule coefficients.
        beta_t = beta[t]
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[t - 1]

        # Posterior variance.
        beta_tilde = (
            beta_t * (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
        )

        # Reverse diffusion mean.
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t -
            (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
        )

        # Sample x_(t-1).
        if t > 1:
            z = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta_tilde) * z
        else:
            x_t = mean

    # Decode final output.
    generated = decode_output(
        generated=x_t,
        config=config,
        vae=vae
    )

    return generated

if __name__ == "__main__":
    import torch
    from model import UNet
    from config import config
    import matplotlib.pyplot as plt
    from modules import NoiseInjectionPipeline
    from denoising_diffusion import NoisePredictionPipeline, DDPM
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    parser = argparse.ArgumentParser(description="DDPM Image Generation Script")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/best.pt", help="Path to the model checkpoint")
    parser.add_argument("--weights", type=str, choices=["ema", "original"], default="ema")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU Execution"

    # Load the best checkpoint
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )
    saved_config = checkpoint.get("config", {})
    active_config = {**config, **saved_config}

    # Validate required configuration.
    if "in_channels" not in active_config:
        active_config["in_channels"] = (
            4 if active_config["latent_denoise"] else 3
        )

    if "out_channels" not in active_config:
        active_config["out_channels"] = (
            4 if active_config["latent_denoise"] else 3
        )

    # Load VAE if latent denoising is enabled
    vae = None

    if active_config["latent_denoise"]:
        vae = AutoencoderKL.from_pretrained(
            active_config["vae_model"]
        ).to(device)
        vae.eval()
        vae.requires_grad_(False)

    # Forward diffusion
    noise_injection_pipeline = NoiseInjectionPipeline(
        config=active_config,
        random=True
    )

    # Noise prediction backbone
    unet = UNet(
        in_channels=active_config["in_channels"],
        out_channels=active_config["out_channels"],
        channel_multipliers=active_config.get(
            "channel_multipliers", (1, 2, 4, 8)
        ),
        time_embd_dim=active_config["time_embed"],
        base_channels=active_config.get("base_channels", 64)
    )

    # Noise prediction pipeline
    noise_prediction_pipeline = NoisePredictionPipeline(
        config=active_config,
        noise_prediction_model=unet
    )

    # Complete DDPM
    ddpm = DDPM(
        noise_injection_pipieline=noise_injection_pipeline,
        noise_prediction_pipeline=noise_prediction_pipeline
    )

    if args.weights == "ema":
        if "ema_model_state_dict" not in checkpoint:
            print("EMA weights unavailable. Falling back to original weights.")
            state_dict = checkpoint["model_state_dict"]
            ema_enabled = False
        else:
            state_dict = checkpoint["ema_model_state_dict"]
            ema_enabled = True
            print("Loaded EMA model weights.")

    else:
        state_dict = checkpoint["model_state_dict"]
        ema_enabled = False
        print("Loaded original model weights.")


    ema_enabled = (args.weights == "ema" and "ema_model_state_dict" in checkpoint)

    ddpm.load_state_dict(state_dict, strict=True)
    ddpm.to(device)
    ddpm.eval()

    print("=" * 60)
    print("DDPM IMAGE GENERATION PIPELINE INITIALIZED")
    print("=" * 60)
    print(f"Execution Device : {device.type.upper()} ({device_name})")
    print(f"Checkpoint Path  : {args.checkpoint}")
    print(f"Loaded From Step : {checkpoint.get('global_step', 'Unknown')}")
    print(f"Checkpoint Loss  : {checkpoint.get('best_val_loss', 'Unknown')}")
    print(f"Latent Denoising : {active_config['latent_denoise']}")
    print(f"Precomputed Latents: {'Enabled' if active_config['precomputed_latents'] else 'Disabled'}")
    print(f"EMA Inference: {'Enabled' if ema_enabled else 'Disabled'}")
    print(f"Input Channels   : {active_config['in_channels']}")
    print(f"Output Channels  : {active_config['out_channels']}")
    print("-" * 60)
    print("DDPM MODEL CONFIGURATION DETAILS:")
    print(saved_config)
    print("-" * 60)
    print("Sampling and generating image... Please wait.")
    print("=" * 60)

    generated_image = sample(ddpm, active_config, device, args.batch_size, vae = vae)

    # Visualize generated images

    # generated_image: [B, C, H, W]; and the sampling pipeline already handles the output normalization range [0, 1]
    # generated_image: (B, C, H, W) in [0, 1] range

    grid = make_grid(
        generated_image,
        # nrow=min(args.batch_size, 4),
        nrow=8,
        padding=0,
        normalize=False
    )
    # CHW -> HWC
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()