"""Shared helper utilities for the diffusion project."""

import random
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import yaml



def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Persist configuration values to disk."""
    with open(path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Configuration saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration values from disk."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from {path}")
    return config


def plot_sequences(
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    clean: Optional[torch.Tensor] = None,
    path: Optional[str] = None,
) -> None:
    """Create visualization comparing noisy, denoised, and clean trajectories."""
    num_samples = min(4, noisy.shape[0])

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]

        ax.plot(noisy[i].detach().cpu().numpy(), 'gray', label='Noisy', alpha=0.5)

        ax.plot(denoised[i].detach().cpu().numpy(), 'r-', label='Denoised', linewidth=2)

        if clean is not None:
            ax.plot(clean[i].detach().cpu().numpy(), 'b-', label='Clean', linewidth=2, alpha=0.7)

        # Styling
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path}")
    
    plt.show()
