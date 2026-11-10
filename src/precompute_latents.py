import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL

from dataset_loader import CelebADataset


@torch.no_grad()
def precompute_latents(
    dataset,
    vae,
    device,
    output_path,
    batch_size=8,
    num_workers=4
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    vae.eval()
    vae.requires_grad_(False)

    scaling_factor = vae.config.scaling_factor

    downsample_factor = 2 ** (
        len(vae.config.block_out_channels) - 1
    )

    image_size = dataset.transform.transforms[0].size[0]

    latent_size = image_size // downsample_factor
    latent_channels = vae.config.latent_channels

    # Preallocate CPU memory for all latent representations.
    all_latents = torch.empty(
        len(dataset),
        latent_channels,
        latent_size,
        latent_size,
        dtype=torch.float32
    )

    offset = 0

    for images in tqdm(loader, desc="Encoding CelebA"):

        images = images.to(
            device,
            non_blocking=True
        )

        # Encode RGB images.
        posterior = vae.encode(images).latent_dist

        # Deterministic latent representation.
        latents = posterior.mode()

        # Apply VAE scaling factor.
        latents = latents * scaling_factor

        latents = latents.float().cpu()

        batch_size_actual = latents.shape[0]

        all_latents[offset:offset + batch_size_actual] = latents

        offset += batch_size_actual

    # Save latent representations and metadata.
    torch.save(
        {
            "latents": all_latents,
            "scaling_factor": scaling_factor,
            "vae_model": vae.config._name_or_path,
            "image_size": image_size,
            "latent_channels": latent_channels,
            "latent_size": latent_size,
            "num_samples": len(dataset),
        },
        output_path
    )

    print("=" * 50)
    print("LATENT PRECOMPUTATION COMPLETED")
    print("=" * 50)
    print(f"Number of images : {len(dataset)}")
    print(f"Latent shape     : {tuple(all_latents.shape)}")
    print(f"Saved to         : {output_path}")
    print("=" * 50)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = CelebADataset(
        root_dir="../data/celeba/img_align_celeba",
        image_size=256
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(device)

    precompute_latents(
        dataset=dataset,
        vae=vae,
        device=device,
        output_path="../data/celeba/celeba_latents.pt",
        batch_size=8
    )
