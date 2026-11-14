import torch
from .base import BaseSampler
from utils import decode_output

class ImprovedDDPMSampler(BaseSampler):
    def __init__(self, ddpm, config, device, vae=None):
        super().__init__(ddpm, config, device, vae)

        beta_tilde = (
            self.beta[1:] * (1.0 - self.alpha_bar[:-1])
            / (1.0 - self.alpha_bar[1:])
        )

        self.posterior_variance_clipped = torch.cat([
            torch.zeros_like(beta_tilde[:1]),
            beta_tilde[1:2],
            beta_tilde[1:]
        ])

    @torch.no_grad()
    def sample(self, batch_size):

        x_t = self._initialize_noise(batch_size)

        for t in range(self.T, 0, -1):

            t_tensor = torch.full(
                (batch_size,),
                t,
                device=self.device,
                dtype=torch.long
            )

            eps_theta, fraction = self.ddpm.noise_prediction_pipeline(
                x_t, t_tensor
            )

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]

            mean = (
                x_t -
                (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            ) / torch.sqrt(alpha_t)

            beta_tilde_t = self.posterior_variance_clipped[t]

            log_var = (
                fraction * torch.log(beta_t)
                + (1.0 - fraction) * torch.log(beta_tilde_t)
            )

            learned_var = torch.exp(log_var)

            if t > 1:
                x_t = mean + torch.sqrt(learned_var) * torch.randn_like(x_t)
            else:
                x_t = mean

        return decode_output(
            generated=x_t,
            config=self.config,
            vae=self.vae
        )