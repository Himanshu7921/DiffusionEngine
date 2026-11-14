import torch
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, ddpm, config, device, vae=None):
        """
        Defines a BaseSampler class for all Samplers which will be implemented in future
        """
        self.device = torch.device(device)
        self.ddpm = ddpm.to(self.device).eval()
        self.config = config
        self.device = device

        self.vae = vae
        if self.vae is not None:
            self.vae = self.vae.to(self.device).eval()

        self.T = config["T"]
        self.image_size = config["image_size"]
        self.channels = config["in_channels"]
        self.latent_denoise = config["latent_denoise"]

        if self.latent_denoise and vae is None:
            raise ValueError(
                "VAE must be provided when latent denoising is enabled."
            )

        self.injector = ddpm.noise_injection_pipieline.noise_injector

        self.beta = self.injector.beta.to(device)
        self.alpha = self.injector.alpha.to(device)
        self.alpha_bar = self.injector.alpha_bar.to(device)

        self.height, self.width = self._get_sample_dimensions()

    def _get_sample_dimensions(self):

        if not self.latent_denoise:
            return self.image_size, self.image_size

        factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1
        )

        if self.image_size % factor != 0:
            raise ValueError(
                "Image size must be divisible by VAE downsample factor."
            )

        size = self.image_size // factor
        return size, size

    def _initialize_noise(self, batch_size):

        return torch.randn(
            batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device
        )

    @abstractmethod
    @torch.no_grad()
    def sample(self, batch_size):
        pass