import torch
from .base import BaseSampler
from utils import decode_output

class DDPMSampler(BaseSampler):
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

            eps_theta, _ = self.ddpm.noise_prediction_pipeline(
                x_t, t_tensor
            )

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t - 1]

            mean = (
                x_t -
                (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            ) / torch.sqrt(alpha_t)

            beta_tilde = (
                beta_t * (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t)
            )

            if t > 1:
                x_t = mean + torch.sqrt(beta_tilde) * torch.randn_like(x_t)
            else:
                x_t = mean

        return decode_output(
            generated=x_t,
            config=self.config,
            vae=self.vae
        )