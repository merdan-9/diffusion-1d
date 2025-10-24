"""Neural network modules for predicting diffusion noise on 1-D trajectories."""

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Transforms scalar diffusion timesteps into learned embeddings."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        """Initialize embedding parameters."""
        super().__init__()
        pass

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Produce embeddings for the provided diffusion timesteps."""
        pass


class NoisePredictor(nn.Module):
    """Predicts noise residuals from noisy trajectories and time embeddings."""

    def __init__(self, seq_length: int, hidden_dim: int, time_dim: int) -> None:
        """Construct the temporal network backbone."""
        super().__init__()
        pass

    def forward(self, noisy_sequence: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return predicted noise residuals."""
        pass
