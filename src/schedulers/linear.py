import torch
import torch.nn as nn

class LinearNoiseScheduler(nn.Module):
    def __init__(self, config):
        """
        Implements a linear noise scheduler that generates the variance schedule
        (beta_t) used during the forward diffusion process

        - Samples beta values uniformly across a predefined linear range.
        - Gradually increases the variance of Gaussian noise injected into the
        input over diffusion timesteps.
        - Follows the original DDPM noise scheduling strategy.

        Reference:
            Ho et al., "Denoising Diffusion Probabilistic Models" (2020).
            https://arxiv.org/pdf/2006.11239
        """
        super().__init__()
        self.config = config
        self.beta_start = config["beta_start"] # where, beta_end > beta_start means gradually increase the amount of noise added
        self.beta_end = config["beta_end"]
        self.T = config["T"]

        beta = torch.zeros(self.T + 1)
        beta[1:] = torch.linspace(self.beta_start, self.beta_end, self.T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim = 0)

        self.register_buffer("alpha", alpha.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("beta", beta.float())

    def forward(self):
        return (self.alpha, self.beta)