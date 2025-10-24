"""Synthetic sine-wave dataset and data-loading utilities."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SineWaveDataset(Dataset):
    """Dataset wrapping noisy sine-wave trajectories and their clean targets."""

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor) -> None:
        """Store pre-generated trajectories.

        Args:
            sequences: Noisy input sequences shaped [N, T].
            targets: Clean target sequences shaped [N, T].
        """
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        """Return dataset size."""
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single (noisy, clean) trajectory pair."""
        return self.sequences[idx], self.targets[idx]


def generate_sine_wave_batch(
    batch_size: int,
    seq_length: int,
    noise_std: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a batch of noisy/clean sine-wave trajectories."""
    timestamps = torch.linspace(0, 2 * torch.pi, seq_length, device=device)  # [T]

    frequencies = torch.rand(batch_size, 1, device=device) * 4.0 + 1.0  # [B, 1]
    phases = torch.rand(batch_size, 1, device=device) * 2 * torch.pi  # [B, 1]

    clean_sequences = torch.sin(frequencies * timestamps + phases)  # [B, T]
    
    return clean_sequences, clean_sequences


def build_dataloader(
    batch_size: int,
    seq_length: int,
    noise_std: float,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader over synthetic sine trajectories."""
    # Generate dataset
    num_samples = 10000
    sequences, targets = generate_sine_wave_batch(
        batch_size=num_samples,
        seq_length=seq_length,
        noise_std=noise_std,
        device=device,
    )

    # Wrap in Dataset and DataLoader
    dataset = SineWaveDataset(sequences, targets)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
