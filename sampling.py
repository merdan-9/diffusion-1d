"""Inference utilities for generating trajectories from the diffusion model."""

from typing import Optional

import torch

from diffusion import Diffusion1D

from utils import set_seed


def generate_sequences(
    diffusion: Diffusion1D,
    num_samples: int,
    seq_length: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample new sine-like sequences using the trained diffusion model."""
    set_seed(seed or 42)

    # Create shape tuple
    shape = (num_samples, seq_length)

    # Generate samples
    samples = diffusion.sample(shape)

    return samples
