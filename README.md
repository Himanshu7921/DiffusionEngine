# DDPM - PyTorch 

<p align="center">
  <img src="assets/ddpm_architecture.png" alt="DDPM Forward and Reverse Diffusion Process" width="100%">
</p>

## Results

<p align="center">
  <img src="assets/generated_result.png"
       alt="DDPM Generated Result"
       width="100%">
</p>

Implements **Denoising Diffusion Probabilistic Models (DDPM)** based on the paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239).

## How DDPM Works

DDPM consists of two complementary processes:

- **Forward Diffusion:** Gradually adds Gaussian noise to the original data over $T$ timesteps until the sample approaches an isotropic Gaussian distribution.
- **Reverse Diffusion:** Learns to progressively remove noise from a noisy sample, reconstructing the original data distribution using a neural network.

The model is trained to predict the noise added at each timestep:

$$
\mathcal{L}_{simple}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right]
$$

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