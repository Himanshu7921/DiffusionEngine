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

## Sampling

Generate images using a trained checkpoint:

```bash
python sample.py --batch_size 4 --checkpoint ./checkpoint/best.pt
```

The sampling pipeline loads the checkpoint and generates images through iterative reverse diffusion, starting from Gaussian noise.

## Training & Evaluation

The [`train.ipynb`](src/train.ipynb) notebook demonstrates the training workflow, model evaluation, and reverse diffusion sampling.