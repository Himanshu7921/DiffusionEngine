DEBUG = True

config = {
    # Diffusion
    "T": 1000,
    "batch_size": 16,
    "image_size": 256,

    # Diffusion mode
    "latent_denoise": True,
    "precomputed_latents": True,

    # U-Net
    "channel_multipliers": (1, 2, 4, 8),

    # Noise schedule
    "scheduler": "linear",  # ["linear", "cosine"]

    # Linear
    "beta_start": 1e-4,
    "beta_end": 2e-2,

    # Cosine
    "s": 0.008,

    # Sampler
    "sampler": "improved_ddpm", # OPTIONS: [ddpm, improved_ddpm] # NOTE: The UNet's config will adapt itself accordingly
    # if sampler == ddpm: Loss = L_simple
    # if sampler == improved_ddpm = L_simple + lambda * L_vlb

    # Loss
    "lambda_vlb": 0.001,
    "use_min_snr_loss": True,

    # Dataset
    "test_img_dir": "../data/test_images/test_img.png",

    # Timestep embedding
    "time_embed": 512,
}

# Automatically configure U-Net channels
config["in_channels"] = 4 if config["latent_denoise"] else 3
config["out_channels"] = 4 if config["latent_denoise"] else 3
config["vae_model"] = "stabilityai/sd-vae-ft-mse" if config["latent_denoise"] else None
