"""Noise schedule utilities for diffusion timesteps."""

import torch


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Return a linearly spaced beta schedule."""
    pass


def cosine_beta_schedule(timesteps: int) -> torch.Tensor:
    """Return betas following the cosine schedule from Nichol & Dhariwal (2021)."""
    pass


def compute_alphas_cumprod(betas: torch.Tensor) -> torch.Tensor:
    """Compute cumulative product of alphas from betas."""
    pass
