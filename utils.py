"""Shared helper utilities for the diffusion project."""

from typing import Any, Dict, Optional

import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    pass


def save_config(config: Dict[str, Any], path: str) -> None:
    """Persist configuration values to disk."""
    pass


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration values from disk."""
    pass


def plot_sequences(
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    clean: Optional[torch.Tensor] = None,
    path: Optional[str] = None,
) -> None:
    """Create visualization comparing noisy, denoised, and clean trajectories."""
    pass
