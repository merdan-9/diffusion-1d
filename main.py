"""Entry point for training and sampling 1-D diffusion models."""

import os

import torch
from torch import optim

from config import DiffusionConfig
from data import build_dataloader
from diffusion import Diffusion1D
from model import NoisePredictor
from model_unet import UNet1D
from sampling import generate_sequences
from schedules import linear_beta_schedule
from train import DiffusionTrainer
from utils import set_seed, plot_sequences


def create_model(config: DiffusionConfig, device: torch.device):
    """Create noise prediction model based on config.

    Args:
        config: Configuration object with model_type and hyperparameters
        device: Device to place model on (cpu or cuda)

    Returns:
        Model instance (NoisePredictor or UNet1D) on specified device
    """
    if config.model_type == "mlp":
        model = NoisePredictor(
            seq_length=config.seq_length,
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim
        )
        print(f"Created MLP model")
    elif config.model_type == "unet":
        model = UNet1D(
            seq_length=config.seq_length,
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim
        )
        print(f"Created UNet1D model")
    else:
        raise ValueError(f"Unknown model_type: '{config.model_type}'. Must be 'mlp' or 'unet'")

    return model.to(device)


def train_model(config: DiffusionConfig, device: torch.device) -> str:
    """Train a diffusion model and return checkpoint path.

    Args:
        config: Configuration object with hyperparameters
        device: Device to train on (cpu or cuda)

    Returns:
        Path to saved checkpoint
    """
    # Create beta schedule
    betas = linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)

    # Create model
    model = create_model(config, device)

    # Create diffusion object
    diffusion = Diffusion1D(
        network=model,
        timesteps=config.timesteps,
        betas=betas,
        device=device
    )

    # Create dataloader
    dataloader = build_dataloader(config.batch_size, config.seq_length, noise_std=config.noise_std, device=device, shuffle=True)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create trainer
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device
    )

    print("Starting training...")
    for epoch in range(config.num_epochs):
        metrics = trainer.train_epoch()
        loss = metrics['train_loss']
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = "outputs/diffusion_model.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Clear GPU cache after training to free memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return checkpoint_path

def sample_model(config: DiffusionConfig, checkpoint_path: str, device: torch.device) -> None:
    """Generate samples from trained model and visualize.

      Args:
          config: Configuration object with hyperparameters
          checkpoint_path: Path to trained model checkpoint
          device: Device to sample on (cpu or cuda)
      """

    # Create beta schedule
    betas = linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)

    # Create model (must match training architecture)
    model = create_model(config, device)

    # Create diffusion object
    diffusion = Diffusion1D(
        network=model,
        timesteps=config.timesteps,
        betas=betas,
        device=device
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Generating samples...")
    num_samples = 16
    samples = generate_sequences(
        diffusion=diffusion,
        num_samples=num_samples,
        seq_length=config.seq_length,
        device=device,
        seed=42
    )

    noise = torch.randn_like(samples)
    output_path = "outputs/sampled_sequences.png"
    plot_sequences(
        noisy=noise,
        denoised=samples,
        path=output_path
    )
    print(f"Sampled sequences saved to {output_path}")

def main() -> None:
    """Execute the requested pipeline stage."""
    
    # Load configuration
    config = DiffusionConfig()

    # Set random seed
    set_seed(42)

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Train model
    checkpoint_path = train_model(config, device)

    # Sample from trained model
    sample_model(config, checkpoint_path, device)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
