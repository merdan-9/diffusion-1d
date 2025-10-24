"""Training loop utilities for the 1-D diffusion model."""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .diffusion import Diffusion1D


class DiffusionTrainer:
    """Handles optimization, evaluation, and checkpointing."""

    def __init__(
        self,
        diffusion: Diffusion1D,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Store references to training components."""
        pass

    def train_epoch(self) -> Dict[str, float]:
        """Run a single training epoch and return aggregate metrics."""
        pass

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation without gradient updates."""
        pass

    def save_checkpoint(self, path: str) -> None:
        """Persist model and optimizer state to disk."""
        pass
