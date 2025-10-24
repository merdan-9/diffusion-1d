"""Noise schedule utilities for diffusion timesteps."""

import torch


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Return a linearly spaced beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int) -> torch.Tensor:
    """Return betas following the cosine schedule from Nichol & Dhariwal (2021)."""
    pass


def compute_alphas_cumprod(betas: torch.Tensor) -> torch.Tensor:
    """Compute cumulative product of alphas from betas."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


if __name__ == "__main__":
    print("Testing noise schedules...")

    # Test linear beta schedule
    timesteps = 100
    beta_start = 1e-4
    beta_end = 0.02

    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    print(f"\nBetas shape: {betas.shape}")
    print(f"First beta: {betas[0]:.6f} (expected: {beta_start})")
    print(f"Last beta: {betas[-1]:.6f} (expected: {beta_end})")

    # Test compute alphas
    alpha_bar = compute_alphas_cumprod(betas)
    print(f"\nAlpha_bar shape: {alpha_bar.shape}")
    print(f"First alpha_bar: {alpha_bar[0]:.6f} (should be close to 1)")
    print(f"Last alpha_bar: {alpha_bar[-1]:.6f} (should be small)")

    print(f"\nSample values at different timesteps:")
    for t in [0, 25, 50, 75, 99]:
        print(f"  t={t:2d}: beta={betas[t]:.6f}, alpha_bar={alpha_bar[t]:.6f}")

    print("\nTests passed!")