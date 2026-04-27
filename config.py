"""Configuration objects and helpers for the diffusion project."""

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Holds hyperparameters for training and sampling the diffusion model."""

    # Model architecture
    model_type: str = "unet"  # "mlp" or "unet"

    # Data parameters
    seq_length: int = 64
    noise_std: float = 0.1

    # Diffusion schedule
    timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Model hyperparameters (shared by both MLP and UNet)
    hidden_dim: int = 128
    time_dim: int = 128

    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 50
    device: str = "cuda"

    # Optional model-based sine guidance during sampling
    enable_sine_guidance: bool = True
    sine_guidance_scale: float = 0.01
    sine_guidance_start_fraction: float = 0.5
    sine_guidance_amplitude_limit: float = 1.25
    sine_guidance_amplitude_weight: float = 0.1
    sine_guidance_min_frequency: float = 1.0
    sine_guidance_max_frequency: float = 5.0
    sine_guidance_num_steps: int = 1
