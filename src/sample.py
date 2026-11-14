import torch
import argparse
import matplotlib.pyplot as plt

from config import config
from diffusers import AutoencoderKL
from torchvision.utils import make_grid

from model import UNet
from modules import NoiseInjectionPipeline
from denoising_diffusion import NoisePredictionPipeline, DDPM

from samplers import DDPMSampler, ImprovedDDPMSampler


# Available sampling strategies.
SAMPLERS = {
    "ddpm": DDPMSampler,
    "improved_ddpm": ImprovedDDPMSampler,
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="DiffusionEngine Image Generation"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images to generate"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint/best.pt",
        help="Path to the model checkpoint"
    )

    parser.add_argument(
        "--weights",
        type=str,
        choices=["ema", "original"],
        default="ema",
        help="Model weights to load"
    )

    parser.add_argument(
        "--sampler",
        type=str,
        choices=list(SAMPLERS.keys()),
        default=None,
        help="Sampling strategy. Overrides the config value."
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    saved_config = checkpoint.get("config", {})

    active_config = {
        **config,
        **saved_config
    }

    # Validate channel configuration.
    active_config.setdefault(
        "in_channels",
        4 if active_config["latent_denoise"] else 3
    )

    active_config.setdefault(
        "out_channels",
        4 if active_config["latent_denoise"] else 3
    )

    return checkpoint, saved_config, active_config


def load_vae(active_config, device):

    if not active_config["latent_denoise"]:
        return None

    vae = AutoencoderKL.from_pretrained(
        active_config["vae_model"]
    ).to(device)

    vae.eval()
    vae.requires_grad_(False)

    return vae


def load_model(active_config, checkpoint, weights, device):

    # Forward diffusion.
    noise_injection_pipeline = NoiseInjectionPipeline(
        config=active_config,
        random=True
    )

    # Noise prediction backbone.
    unet = UNet(
        in_channels=active_config["in_channels"],
        out_channels=active_config["out_channels"],
        channel_multipliers=active_config.get(
            "channel_multipliers", (1, 2, 4, 8)
        ),
        time_embd_dim=active_config["time_embed"],
        base_channels=active_config.get("base_channels", 64),
        learned_fraction= config["sampler"].lower() == "improved_ddpm"
    )

    # Noise prediction pipeline.
    noise_prediction_pipeline = NoisePredictionPipeline(
        config=active_config,
        noise_prediction_model=unet
    )

    # Complete DDPM.
    ddpm = DDPM(
        noise_injection_pipieline=noise_injection_pipeline,
        noise_prediction_pipeline=noise_prediction_pipeline
    )

    # Select model weights.
    if weights == "ema" and "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
        ema_enabled = True
        print("Loaded EMA model weights.")

    else:
        if weights == "ema":
            print("EMA weights unavailable. Falling back to original weights.")
        state_dict = checkpoint["model_state_dict"]
        ema_enabled = False
        print("Loaded original model weights.")

    # Remove redundant scheduler buffers from older checkpoints.
    state_dict = { # i have to do this, because i modified the NoiseInjector and the previously saved checkpoint contains it
        key: value
        for key, value in state_dict.items()
        if ".noise_injector.scheduler." not in key
        and ".noise_injector.noise_scheduler." not in key
    }

    ddpm.load_state_dict(state_dict, strict=True)

    ddpm.to(device)
    ddpm.eval()

    return ddpm, ema_enabled


def load_sampler(sampler_name, ddpm, active_config, device, vae=None):
    if sampler_name not in SAMPLERS:
        raise ValueError(
            f"Unknown sampler: {sampler_name}. "
            f"Available samplers: {list(SAMPLERS.keys())}"
        )

    sampler_class = SAMPLERS[sampler_name]

    sampler = sampler_class(
        ddpm=ddpm,
        config=active_config,
        device=device,
        vae=vae
    )

    return sampler

@torch.no_grad()
def generate(sampler, batch_size):
    return sampler.sample(batch_size = batch_size)


def visualize(generated_images, batch_size):

    # generated_images: [B, C, H, W], normalized to [0, 1].
    grid = make_grid(
        generated_images,
        nrow=8,
        padding=0,
        normalize=False
    )

    # CHW -> HWC.
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis("off")

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    plt.show()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU Execution"
    )

    # Load checkpoint and configuration.
    checkpoint, saved_config, active_config = load_checkpoint(
        args.checkpoint,
        device
    )

    # sampling strategy
    sampler_name = (
        args.sampler
        if args.sampler is not None
        else active_config.get("sampler", "improved_ddpm")
    )

    if sampler_name not in SAMPLERS:
        raise ValueError(
            f"Invalid sampler '{sampler_name}'. "
            f"Available: {list(SAMPLERS.keys())}"
        )

    # Load VAE.
    vae = load_vae(active_config, device)

    # Load DDPM model.
    ddpm, ema_enabled = load_model(
        active_config=active_config,
        checkpoint=checkpoint,
        weights=args.weights,
        device=device
    )

    # Dynamically initialize sampler.
    sampler = load_sampler(
        sampler_name=sampler_name,
        ddpm=ddpm,
        active_config=active_config,
        device=device,
        vae=vae
    )

    output_channels = active_config["out_channels"]
    if sampler_name == "improved_ddpm":
        output_channels *= 2

    # Display configuration.
    print("=" * 60)
    print("DIFFUSIONENGINE IMAGE GENERATION PIPELINE")
    print("=" * 60)

    print(f"Execution Device    : {device.type.upper()} ({device_name})")
    print(f"Checkpoint Path     : {args.checkpoint}")
    print(f"Loaded From Step    : {checkpoint.get('global_step', 'Unknown')}")
    print(f"Scheduler           : {active_config.get('scheduler', 'Unknown')}")
    print(f"Checkpoint Loss     : {checkpoint.get('best_val_loss', 'Unknown')}")
    print(f"Sampling Strategy   : {sampler_name}")
    print(f"Latent Denoising    : {active_config['latent_denoise']}")
    print(f"Precomputed Latents : {'Enabled' if active_config.get('precomputed_latents', False) else 'Disabled'}")
    print(f"EMA Inference       : {'Enabled' if ema_enabled else 'Disabled'}")
    print(f"Input Channels      : {active_config['in_channels']}")
    print(f"Output Channels     : {output_channels}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Timesteps           : {active_config['T']}")

    print("-" * 60)
    print("MODEL CONFIGURATION:")
    print(saved_config)

    print("-" * 60)
    print("Generating images. Please wait...")
    print("=" * 60)

    # Generate images.
    generated_images = generate(
        sampler=sampler,
        batch_size=args.batch_size
    )

    print("Image generation completed.")

    # Visualize generated images.
    visualize(
        generated_images=generated_images,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
