"""Configuration objects and helpers for the diffusion project."""

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Holds hyperparameters for training and sampling the diffusion model."""

    seq_length: int = 64
    timesteps: int = 1000
    beta_schedule: str = "linear"
    noise_std: float = 0.1
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda"
