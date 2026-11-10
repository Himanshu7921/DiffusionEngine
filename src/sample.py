import torch
import argparse

@torch.no_grad()
def sample(ddpm, config, device, batch_size):
    ddpm.eval()

    T = config["T"]

    # Start from Gaussian noise with the same shape as the training image.
    x_t = torch.randn(batch_size, 3, 64, 64).to(device)

    injector = ddpm.noise_injection_pipieline.noise_injector

    # Move schedules to the sampling device.
    beta = injector.beta.to(device)
    alpha = injector.alpha.to(device)
    alpha_bar = injector.alpha_bar.to(device)

    for t in range(T, 0, -1):

        t_tensor = torch.tensor([t], device=device, dtype=torch.long)

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

    return x_t

if __name__ == "__main__":
    import torch
    from model import UNet
    from config import config
    import matplotlib.pyplot as plt
    from modules import NoiseInjectionPipeline
    from denoising_diffusion import NoisePredictionPipeline, DDPM

    parser = argparse.ArgumentParser(description="DDPM Image Generation Script")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/best.pt", help="Path to the model checkpoint")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Determine active device name
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU Execution"


    # Forward diffusion
    noise_injection_pipeline = NoiseInjectionPipeline(
        config=config,
        random=True
    )

    # Noise prediction backbone
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_embd_dim=config["time_embed"],
        base_channels=64
    )

    # Noise prediction pipeline
    noise_prediction_pipeline = NoisePredictionPipeline(
        config=config,
        noise_prediction_model=unet
    )

    # Complete DDPM
    ddpm = DDPM(
        noise_injection_pipieline=noise_injection_pipeline,
        noise_prediction_pipeline=noise_prediction_pipeline
    )

    # Load the best checkpoint
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    # Restore model weights
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    ddpm.to(device)
    ddpm.eval()

    # 2. Extract configuration info from the loaded checkpoint if available
    saved_config = checkpoint.get("config", "No config metadata found in checkpoint")

    print("=" * 60)
    print("DDPM IMAGE GENERATION PIPELINE INITIALIZED")
    print("=" * 60)
    print(f"Execution Device : {device.type.upper()} ({device_name})")
    print(f"Checkpoint Path  : {args.checkpoint}")
    print(f"Loaded From Step : {checkpoint.get('global_step', 'Unknown')}")
    print(f"Checkpoint Loss  : {checkpoint.get('best_val_loss', 'Unknown')}")
    print("-" * 60)
    print("DDPM MODEL CONFIGURATION DETAILS:")
    print(saved_config)
    print("-" * 60)
    print("Sampling and generating image... Please wait.")
    print("=" * 60)

    generated_image = sample(ddpm, config, device, args.batch_size)
    generated_image = (generated_image + 1.0) / 2.0 # Convert [-1, 1] -> [0, 1]
    generated_image = generated_image.clamp(0, 1) # for visualization in matplotlib

    import math

    # Get the total number of images in the batch (B)
    num_images = generated_image.shape[0]

    # Check if we need to enforce the exact 10x10 zero-gap matrix layout
    if num_images == 100:
        rows, cols = 10, 10
        # Set up the strict 10x10 grid with absolutely zero spacing
        fig, axes = plt.subplots(
            rows, 
            cols, 
            figsize=(10, 10), 
            gridspec_kw={'wspace': 0, 'hspace': 0}
        )
    else:
        # Fallback to your dynamic calculation for other batch sizes
        cols = min(4, num_images)  # Max 4 images per row
        rows = math.ceil(num_images / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Ensure axes is always a flat array even if rows/cols = 1
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_images):
        # Convert BCHW -> HWC for the i-th image
        img = generated_image[i].permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(img)
        axes[i].axis("off")
        
        # Only show image titles if it is NOT the 100-image grid layout
        if num_images != 100:
            axes[i].set_title(f"Image {i+1}")

    # Hide any empty subplots if the grid isn't perfectly filled
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    # Apply layout adjustment based on the batch type
    if num_images == 100:
        # Force the bounding box precisely to the edges with zero white borders
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    else:
        plt.tight_layout()

    plt.show()

