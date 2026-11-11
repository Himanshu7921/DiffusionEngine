import argparse

from config import config
from model import UNet
from modules import NoiseInjectionPipeline
from denoising_diffusion import NoisePredictionPipeline, DDPM
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DDPM model using DiffusionEngine."
    )

    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )

    # Training
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    # Model
    parser.add_argument("--base_channels", type=int, default=64)

    # Optimization
    parser.add_argument(
        "--use_min_snr_loss",
        action=argparse.BooleanOptionalAction,
        default=None
    )

    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=None
    )

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    # Experiment tracking
    parser.add_argument("--project_name", type=str, default="DiffusionEngine")
    parser.add_argument("--run_name", type=str, default="DDPM-Experiment")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--validation_split", type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve configuration
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config["batch_size"]
    )

    use_min_snr_loss = (
        args.use_min_snr_loss
        if args.use_min_snr_loss is not None
        else config["use_min_snr_loss"]
    )

    use_amp = (
        args.use_amp
        if args.use_amp is not None
        else config.get("use_amp", False)
    )

    # Forward diffusion
    noise_injection_pipeline = NoiseInjectionPipeline(
        config=config,
        random=True
    )

    # Noise prediction backbone
    unet = UNet(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        channel_multipliers=(1, 2, 4, 8),
        time_embd_dim=config["time_embed"],
        base_channels=args.base_channels
    )

    # Noise prediction pipeline
    noise_prediction_pipeline = NoisePredictionPipeline(
        config=config,
        noise_prediction_model=unet
    )

    # Complete DDPM
    ddpm = DDPM(
        noise_injection_pipieline=noise_injection_pipeline,
        noise_prediction_pipeline=noise_prediction_pipeline
    )

    # Trainer
    trainer = Trainer(
        ddpm=ddpm,
        config=config,

        dataset_path=args.dataset_path,

        batch_size=batch_size,
        image_size=config["image_size"],

        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        learning_rate=args.learning_rate,
        use_min_snr_loss=use_min_snr_loss,

        validation_split=args.validation_split,

        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,

        project_name=args.project_name,
        run_name=args.run_name,

        num_workers=args.num_workers,
        use_amp=use_amp
    )

    # Resume training if requested
    if args.resume:
        trainer.resume(checkpoint_path=args.resume)

    # Start training
    trainer.fit()


if __name__ == "__main__":
    main()
