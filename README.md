# DDPM - PyTorch

<p align="center">
  <img src="assets/ddpm_architecture.png" alt="DDPM Forward and Reverse Diffusion Process" width="100%">
</p>

Implements **Denoising Diffusion Probabilistic Models (DDPM)** based on the paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239).

## Results

<p align="center">
  <img src="assets/generated_result.png" alt="DDPM Generated Results" width="100%">
</p>

## Features

- Modular DDPM implementation with configurable noise scheduling.
- U-Net noise prediction backbone with timestep embeddings.
- Forward and reverse diffusion processes.
- Configurable pixel-space and latent-space diffusion.
- Pretrained VAE integration using Hugging Face Diffusers.
- Training and sampling support for both diffusion modes.
- Checkpoint-based model restoration and image generation.

## Usage

### Configuration

The diffusion pipeline is controlled through `config.py`.

```python
config = {
    "T": 1000,
    "batch_size": 32,

    # Diffusion mode
    "latent_denoise": False,

    # Noise schedule
    "beta_start": 1e-4,
    "beta_end": 2e-2,

    # Dataset
    "image_size": 256,

    # VAE
    "vae_model": "stabilityai/sd-vae-ft-mse",

    # Timestep embedding
    "time_embed": 256,
}

# Automatically determine U-Net channel dimensions
config["in_channels"] = 4 if config["latent_denoise"] else 3
config["out_channels"] = 4 if config["latent_denoise"] else 3
```

### Initialize the DDPM pipeline

```python
# Initialize the forward diffusion process
noise_injection_pipeline = NoiseInjectionPipeline(
    config=config,
    random=True
)

# Initialize the U-Net noise prediction backbone
unet = UNet(
    in_channels=config["in_channels"],
    out_channels=config["out_channels"],
    time_embd_dim=config["time_embed"],
    base_channels=64
)

# Initialize the noise prediction pipeline
noise_prediction_pipeline = NoisePredictionPipeline(
    config=config,
    noise_prediction_model=unet
)

# Assemble the complete DDPM pipeline
ddpm = DDPM(
    noise_injection_pipieline=noise_injection_pipeline,
    noise_prediction_pipeline=noise_prediction_pipeline
)
```

## Pixel-Space Diffusion

By default, DDPM operates directly on RGB images.

Set:

```python
config["latent_denoise"] = False
```

In this mode, the U-Net receives and predicts noise in RGB image space.

For a 256 × 256 input image:

- Input: `(B, 3, 256, 256)`
- Output: `(B, 3, 256, 256)`

## Latent-Space Diffusion

Training DDPM directly on high-resolution images can be computationally expensive. To address this, the implementation supports **latent-space denoising** using a pretrained Variational Autoencoder (VAE).

The VAE encodes RGB images into a lower-dimensional latent representation. DDPM then learns to predict and remove noise directly in this latent space.

### Pretrained VAE

This implementation uses Hugging Face Diffusers' `AutoencoderKL` with the pretrained checkpoint:

[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)

Install the required dependencies:

```bash
pip install diffusers transformers accelerate safetensors
```

### Enable latent-space diffusion

Set the following configuration:

```python
config["latent_denoise"] = True
```

The U-Net channel dimensions are automatically configured:

```python
config["in_channels"] = 4
config["out_channels"] = 4
```

The pretrained VAE is loaded automatically by the `Trainer` and sampling pipeline when latent denoising is enabled.

### How it works

For a 256 × 256 RGB image:

| Stage | Tensor shape |
|---|---|
| Original image | `(B, 3, 256, 256)` |
| VAE encoder output | `(B, 4, 32, 32)` |
| DDPM input | `(B, 4, 32, 32)` |
| U-Net noise prediction | `(B, 4, 32, 32)` |
| VAE decoder output | `(B, 3, 256, 256)` |

The VAE applies a spatial downsampling factor of 8.

The latent representation is scaled using the VAE's configured `scaling_factor` before diffusion and unscaled before decoding.

### Training support

The `Trainer` class in `train.py` supports both pixel-space and latent-space training through the `latent_denoise` configuration flag.

When enabled:

1. The pretrained VAE is loaded and frozen.
2. Input RGB images are encoded into latent representations.
3. The diffusion process injects noise into the latent representations.
4. The U-Net learns to predict noise in latent space.

When disabled, the Trainer operates directly on RGB images without loading the VAE.

No changes to the training workflow are required beyond configuring `latent_denoise`.

### Sampling support

The `sample()` function in `sample.py` also supports both modes.

It automatically determines the initial noise dimensions from the saved configuration:

- Pixel-space: `(B, 3, 256, 256)`
- Latent-space: `(B, 4, 32, 32)`

After reverse diffusion, when latent denoising is enabled, the generated latent representations are decoded through the pretrained VAE into RGB images.

The sampling function returns images in the `[0, 1]` range in both modes.

## Training & Evaluation

The `Trainer` class in `train.py` handles the training workflow, including:

- Dataset loading and batching.
- Forward diffusion and noise prediction.
- Training and validation.
- Loss tracking and logging.
- Checkpoint saving and restoration.

The diffusion mode is determined by `config["latent_denoise"]`.

## Sampling

Generate images using a trained checkpoint:

```bash
python sample.py --batch_size 4 --checkpoint ./checkpoint/best.pt
```

The sampling pipeline restores the model configuration from the checkpoint and automatically selects the appropriate diffusion mode.

No additional command-line argument is required to enable latent-space sampling.

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/pdf/2006.11239)
- [Hugging Face Diffusers - AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
- [Stable Diffusion VAE - sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
