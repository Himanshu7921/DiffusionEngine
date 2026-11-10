import torch.nn as nn
import torch
from modules import NoiseInjectionPipeline
from model import NoiseModel, UNet
# from config import config
from utils import load_test_img

class NoisePredictionPipeline(nn.Module):
    def __init__(self, config, noise_prediction_model: NoiseModel):
        super().__init__()
        self.T = config["T"]
        self.time_embed = config["time_embed"]
        self.noise_model = noise_prediction_model

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        x_t.shape = (B, C, H, W)
        t.shape = (B, )
        """
        epsilion = self.noise_model(x_t, t) # epsilion.shape = (B, C, H, W)
        return epsilion
    
class DDPM(nn.Module):
    def __init__(self, noise_injection_pipieline: NoiseInjectionPipeline, noise_prediction_pipeline: NoisePredictionPipeline):
        super().__init__()
        self.noise_injection_pipieline = noise_injection_pipieline # forward diffusion
        self.noise_prediction_pipeline = noise_prediction_pipeline # noise prediction model

    def forward(self, x_0: torch.Tensor, timesteps = None, noise = None):
        """
        x_0:      (B, C, H, W)
        timesteps: (B,) or None
        noise:     (B, C, H, W) or None

        If timesteps/noise are provided, reuse them
        Otherwise, generate random values


        timesteps and noise = None (normal mode) | fixed at validation time (makes LRScheuler less sensitive on random noise)
    """
        (x_t, actual_noise), t = self.noise_injection_pipieline(x_0, timesteps = timesteps, noise = noise)
        self.t = t
        pred_noise = self.noise_prediction_pipeline(x_t, t)
        return pred_noise, actual_noise # ([B, C, H, W], [B, C, H, W])


# if __name__ == "__main__":
#     noise_injection_pipeline = NoiseInjectionPipeline(config = config)

#     unet = UNet(
#         in_channels = 3,
#         out_channels = 3,
#         time_embd_dim = 64,
#         base_channels = 64
#     )

#     # wrap u-net into Noise Prediction Pipeline
#     noise_prediction_pipeline = NoisePredictionPipeline(
#         config = config,
#         noise_prediction_model = unet
#     )

#     # Get Noise Injector (uses linear Scheduler)
#     noise_injection_pipeline = NoiseInjectionPipeline(config = config, random = True)

#     ddpm = DDPM(
#         noise_injection_pipieline = noise_injection_pipeline,
#         noise_prediction_pipeline = noise_prediction_pipeline
#     )
    
#     x_0 = load_test_img()
