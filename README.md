# DDPM - PyTorch Implementation

Implements **Denoising Diffusion Probabilistic Models (DDPM)** based on the paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239).

## Usage

```python
# Initialize the forward diffusion process (linear noise scheduler)
noise_injection_pipeline = NoiseInjectionPipeline(
    config=config,
    random=True
)

# Initialize the U-Net noise prediction backbone
unet = UNet(
    in_channels=3,
    out_channels=3,
    time_embd_dim=64,
    base_channels=64
)

# Wrap the U-Net into the noise prediction pipeline
noise_prediction_pipeline = NoisePredictionPipeline(
    config=config,
    noise_prediction_model=unet
)

# Assemble the complete DDPM pipeline
ddpm = DDPM(
    noise_injection_pipieline=noise_injection_pipeline,
    noise_prediction_pipeline=noise_prediction_pipeline
)

# Load and preprocess the input image
x_0 = load_test_img()
```

## Training & Evaluation

The [`train.ipynb`](train.ipynb) notebook demonstrates the training workflow, model evaluation, and reverse diffusion sampling.