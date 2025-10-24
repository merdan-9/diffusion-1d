"""Synthetic sine-wave dataset and data-loading utilities."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SineWaveDataset(Dataset):
    """Dataset wrapping noisy sine-wave trajectories and their clean targets."""

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor) -> None:
        """Store pre-generated trajectories.

        Args:
            sequences: Noisy input sequences shaped [N, T, 1].
            targets: Clean target sequences shaped [N, T, 1].
        """
        pass

    def __len__(self) -> int:
        """Return dataset size."""
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single (noisy, clean) trajectory pair."""
        pass


def generate_sine_wave_batch(
    batch_size: int,
    seq_length: int,
    noise_std: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of noisy/clean sine-wave trajectories."""
    pass


def build_dataloader(
    batch_size: int,
    seq_length: int,
    noise_std: float,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader over synthetic sine trajectories."""
    pass
