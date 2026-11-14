import torch
import torch.nn as nn

class CosineNoiseScheduler(nn.Module):
    def __init__(self, config):
        """
        Implements a cosine noise scheduler that generates the variance schedule
        (beta_t) used during the forward diffusion process.

        - Constructs a cosine-based noise schedule using a cosine cumulative
        noise schedule (alpha_bar_t).
        - Gradually adjusts the noise variance across diffusion timesteps
        according to a cosine function.
        - Computes beta_t from consecutive cumulative alpha values.
        - Applies clipping to beta_t to maintain numerical stability.
        - Follows the improved DDPM noise scheduling strategy.

        Reference:
            Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021).
            https://arxiv.org/pdf/2102.09672

        NOTE:
        Since 'DiffusionEngine's RandomTimeStepSampler' produces timesteps from torch.rand(1, T+1) per batch,
        we need to account for alpha, beta and alpha_bar to have shape = (T+1, )
        
        - Existing LinearScheduler already handles it, and all the downstream layers needs (T+1) shaped tensor
        - This implementation respects that decision and inherits it so downstream layer won't have any effect
        """
        super().__init__()
        self.config = config
        self.T = config["T"]
        self.s = config["s"]

        # Generate timesteps: [0, 1, ..., T]
        self.t = torch.arange(0, self.T + 1, dtype = torch.float64) # get time timesteps shape: (T+1, )

        # compute cumulative alpha's for cosine scheduler
        f_t = self.__cosine_fn(t = self.t)
        f_0 = f_t[0]
        alpha_bar = f_t / f_0 # normalize so alpha_bar becomes '1'; alpha_0: how much information to retain from original image 
        # alpha_bar.shape = (T+1, )

        # Derive beta from consecutive cumulative alpha values
        beta = 1.0 - (alpha_bar[1:] / alpha_bar[:-1]) # beta.shape = (T, )
        beta = beta.clamp(max = 0.999) # clip to prevent singularities at the end of the diffusion process near t = T
        
        # Add the initial no-noise transition
        beta = torch.cat([torch.zeros(1, dtype=beta.dtype), beta]) # (T, ) --> (T+1, )
        alpha = 1.0 - beta # alpha.shape = (T, ) --> (T+1, )

        alpha_bar = torch.cumprod(alpha, dim = 0) # recompute alpha_bar from the clipped beta values Shape: (T, ) --> (T+1, )

        self.register_buffer("alpha", alpha.float()) # (T+1, )
        self.register_buffer("alpha_bar", alpha_bar.float()) # (T+1, )
        self.register_buffer("beta", beta.float()) # (T+1, )

    def __cosine_fn(self, t: torch.Tensor):
        """
        Computes the Cosine scheduler function proposed in the above paper
        """
        f_t = torch.cos((((t/self.T) + self.s) / (1 + self.s)) * (torch.pi/2))
        return f_t ** 2

    def forward(self):
        return (self.alpha, self.beta)