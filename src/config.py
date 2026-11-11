DEBUG = True

config = {
    "T": 1000,
    "batch_size": 16,

    # Diffusion mode
    "latent_denoise": True,
    "precomputed_latents": True,

    "image_size": 256,

    # Noise schedule
    "beta_start": 1e-4,
    "beta_end": 2e-2,

    "use_min_snr_loss": True,

    # Dataset
    # "test_img_dir": "../data/test_images/celeba_02.png",

    # Attention
    # "q_dim": 64,
    # "v_dim": 64,

    # Timestep embedding
    "time_embed": 256,
}

# Automatically configure U-Net channels
config["in_channels"] = 4 if config["latent_denoise"] else 3
config["out_channels"] = 4 if config["latent_denoise"] else 3
config["vae_model"] = "stabilityai/sd-vae-ft-mse" if config["latent_denoise"] else None