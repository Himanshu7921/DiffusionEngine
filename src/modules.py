import torch
import torch.nn as nn
from schedulers import LinearNoiseScheduler, CosineNoiseScheduler

class RandomTimeStepSampler(nn.Module):
    """
    Gives a random TimeStep for the Noise Prediction Model to train on
    """
    def __init__(self, config):
        super().__init__()
        self.T = config["T"]
    
    def forward(self, batch_size: int, device = None):
        """
        returns randomly sampled timestep tensors one per batch
        return: (batch_size,) 
        """
        return torch.randint(low = 1, high = self.T + 1, size = (batch_size, ), dtype = torch.long, device = device)

class DeterministicTimeStepSampler(nn.Module):
    """
    Gives a deterministic TimeStep for Experimentation Purpose
    """
    def __init__(self, config):
        super().__init__()
        self.T_test = config["T_test"]
        self.batch_size = config["batch_size"]
    
    def forward(self, device = None):
        return torch.ones(self.batch_size, dtype = torch.long, device = device) * self.T_test

class NoiseInjector(nn.Module):
    """
    Injects the noise upto the timestep 't' to the image x_0

    training time: Random noise and timestep injection
    validation time: Fixed noise and timestep injection (if provided externally)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.noise_scheduler = self.__get_noise_scheduler(config)
        alpha, beta = self.noise_scheduler() # alpha.shape = [T+1], beta.shape = [T+1], T: Total timestep
        alpha_bar = self.noise_scheduler.alpha_bar # alpha_bar.shape = [T+1]
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta", beta)

    def __get_noise_scheduler(self, config):
        if config["scheduler"] == "linear".lower():
            self.scheduler = LinearNoiseScheduler(config = config)
        elif config["scheduler"] == "cosine".lower():
            self.scheduler = CosineNoiseScheduler(config = config)
        return self.scheduler

    def forward(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        x_0: Original Image at timestep '0'     shape = [B, C, H, W]
        t: Timestep up to which we want to inject the noise shape = [B, ]

        timesteps and noise = None (normal mode) | fixed at validation time (makes LRScheuler less sensitive on random noise)
        """
        B, C, H, W = x_0.shape
        # Generate noise only if not provided.
        if noise is None:
            noise = torch.randn_like(x_0) # get the gaussian noise
        eps = noise
        self.alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1) # (B, 1, 1, 1)
        noisy_image = torch.sqrt(self.alpha_bar_t) * x_0 + torch.sqrt((1 - self.alpha_bar_t)) * eps
        return noisy_image, eps

class NoiseInjectionPipeline(nn.Module):
    def __init__(self, config, random: bool = True):
        """
        Randomly generates the Timesteps for training and uses a fixed noise and timesteps for validation (if provided externally)
        """
        super().__init__()
        self.random = random
        if random:
            self.time_step_sampler = RandomTimeStepSampler(config = config)
        else:
            self.time_step_sampler = DeterministicTimeStepSampler(config = config)
        
        self.noise_injector = NoiseInjector(config = config)

    def forward(self, x_0: torch.tensor, timesteps: torch.Tensor = None, noise: torch.Tensor = None):
        """
        x_0: (B, C, H, W)
        timesteps: (B, )
        timesteps and noise = None (normal mode) | fixed at validation time (makes LRScheuler less sensitive on random noise)
        """
        B = x_0.shape[0]

        if timesteps is None:
            if self.random:
                t = self.time_step_sampler(batch_size = B, device = x_0.device)
            else:
                t = self.time_step_sampler(device = x_0.device)
        else:
            t = timesteps.to(
                device=x_0.device,
                dtype=torch.long
            )

        # Inject supplied noise or generate random noise
        x_t, actual_noise = self.noise_injector(x_0 = x_0, t = t, noise = noise)
        return (x_t, actual_noise), t