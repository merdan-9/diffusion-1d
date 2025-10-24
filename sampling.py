"""Inference utilities for generating trajectories from the diffusion model."""

from typing import Optional

import torch

from .diffusion import Diffusion1D


def generate_sequences(
    diffusion: Diffusion1D,
    num_samples: int,
    seq_length: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample new sine-like sequences using the trained diffusion model."""
    pass
