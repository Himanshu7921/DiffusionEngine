from config import config
import torch

def load_test_img():
    from PIL import Image
    from torchvision import transforms

    image = Image.open(config["test_img_dir"]).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL image -> Tensor [0, 1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [0, 1] -> [-1, 1]
    ])

    x_0 = transform(image)
    x_0 = x_0.unsqueeze(0)

    return x_0

@torch.no_grad()
def decode_output(generated, config, vae=None):
    if config["latent_denoise"]:
        if vae is None:
            raise ValueError(
                "VAE must be provided when latent_denoise=True"
            )

        generated = generated / vae.config.scaling_factor
        generated = vae.decode(generated).sample

    return ((generated + 1) / 2).clamp(0, 1)