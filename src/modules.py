import torch
import torch.nn as nn

class RandomTimeStepSampler(nn.Module):
    """
    Gives a random TimeStep for the Noise Prediction Model to train on
    """
    def __init__(self, config):
        super().__init__()
        self.T = config["T"]
    
    def forward(self, batch_size: int, device = None):
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

class LinearNoiseScheduler(nn.Module):
    """
    This Linear Noise Scheduler gives the noise, 'beta_t', injected during noise injection
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.beta_start = config["beta_start"] # where, beta_end > beta_start means gradually increase the amount of noise added
        self.beta_end = config["beta_end"]
        self.T = config["T"]

        beta = torch.zeros(self.T + 1)
        beta[1:] = torch.linspace(self.beta_start, self.beta_end, self.T)

        alpha = 1 - beta

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)

    def forward(self):
        return (self.alpha, self.beta)

class NoiseInjector(nn.Module):
    """
    Injects the noise upto the timestep 't' to the image x_0
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_noise_scheduler = LinearNoiseScheduler(config = config)
        self.alpha, self.beta = self.linear_noise_scheduler() # # alpha.shape = [T+1], beta.shape = [T+1], T: Total timestep
        alpha_bar = torch.cumprod(self.alpha, dim = 0) # alpha_bar.shape = [T+1]
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        x_0: Original Image at timestep '0'     shape = [B, C, H, W]
        t: Timestep up to which we want to inject the noise shape = [B, ]
        """
        B, C, H, W = x_0.shape
        eps = torch.randn_like(x_0) # get the gaussian noise
        self.alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1) # (B, 1, 1, 1)
        noisy_image = torch.clip((torch.sqrt(self.alpha_bar_t) * x_0 + torch.sqrt((1 - self.alpha_bar_t)) * eps), min = 0, max = 1)
        return noisy_image, eps

class NoiseInjectionPipeline(nn.Module):
    def __init__(self, config, random: bool = True):
        super().__init__()
        self.random = random
        if random:
            self.time_step_sampler = RandomTimeStepSampler(config = config)
        else:
            self.time_step_sampler = DeterministicTimeStepSampler(config = config)
        
        self.noise_injector = NoiseInjector(config = config)

    def forward(self, x_0: torch.tensor):
        B = x_0.shape[0]
        if self.random:
            t = self.time_step_sampler(batch_size = B, device = x_0.device)
        else:
            t = self.time_step_sampler(device = x_0.device)
        return self.noise_injector(x_0, t), t        