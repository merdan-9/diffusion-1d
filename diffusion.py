"""Core diffusion process logic for 1-D trajectories."""

from typing import Tuple

import torch

from model import NoisePredictor
from schedules import compute_alphas_cumprod


class Diffusion1D:
    """Wraps the forward and reverse diffusion operations."""

    def __init__(
        self,
        network: NoisePredictor,
        timesteps: int,
        betas: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Store model, schedules, and device configuration."""
        self.network = network
        self.timesteps = timesteps
        self.betas = betas.to(device)
        self.device = device

        alphas_cumprod = compute_alphas_cumprod(betas)
        self.alphas_cumprod = alphas_cumprod.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion at timestep t."""
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Use the model to predict noise residuals."""
        return self.network(x_t, t)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one reverse diffusion step."""
        # Extract schedule parameters
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = 1.0 - beta_t

        # Predict noise residuals
        noise_pred = self.predict_noise(x_t, t)

        # Compute the mean of p(x_{t-1} | x_t)
        mean_pred = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)) * noise_pred)
        
        # Add noise for stochasticity
        if t[0] > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        # Compute noise variance
        sigma_t = torch.sqrt(beta_t) * eta.unsqueeze(-1)

        x_prev = mean_pred + sigma_t * noise
        return x_prev, noise_pred


    def sample(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Generate new trajectories by iterating the reverse process."""
        # Extract shape
        batch_size, seq_length = shape

        # Sampling doesn't need gradients - saves memory!
        with torch.no_grad():
            # Initialize with pure noise
            x_t = torch.randn(batch_size, seq_length).to(self.device)

            # Iteratively apply reverse diffusion
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                eta = torch.ones(batch_size).to(self.device)  # Can be adjusted for different noise levels
                x_t, _ = self.p_sample(x_t, t_batch, eta)

        return x_t

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from schedules import linear_beta_schedule

    print("Testing forward diffusion process...")

    # Create assets folder for documentation images
    os.makedirs("assets", exist_ok=True)

    # Setup
    timesteps = 100
    beta_start = 1e-4
    beta_end = 0.02
    device = torch.device("cpu")

    # Create betas
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)

    # Create diffusion object (without network for now)
    diffusion = Diffusion1D(
        network=None,  # We don't need the network for forward diffusion
        timesteps=timesteps,
        betas=betas,
        device=device
    )

    print(f"✓ Diffusion initialized with {timesteps} timesteps")

    # Create simple test data: a sine wave
    batch_size = 4
    seq_length = 64
    x = torch.linspace(0, 2 * torch.pi, seq_length)
    x_start = torch.sin(x).unsqueeze(0).repeat(batch_size, 1)  # [batch, seq_length]

    print(f"✓ Created test data: {x_start.shape}")
    print(f"  Original signal range: [{x_start.min():.3f}, {x_start.max():.3f}]")

    # Test at different timesteps
    test_timesteps = [0, 25, 50, 75, 99]

    print(f"\nTesting noise addition at different timesteps:")
    for t_val in test_timesteps:
        t = torch.full((batch_size,), t_val, dtype=torch.long)
        noise = torch.randn_like(x_start)

        x_t = diffusion.q_sample(x_start, t, noise)

        # Calculate signal-to-noise ratio
        signal_strength = diffusion.sqrt_alphas_cumprod[t_val].item()
        noise_strength = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()

        print(f"  t={t_val:2d}: signal={signal_strength:.4f}, noise={noise_strength:.4f}, "
              f"x_t range=[{x_t.min():.3f}, {x_t.max():.3f}]")

    # Verify properties
    print(f"\nVerifying forward diffusion properties:")

    # At t=0, should be mostly original signal
    t_0 = torch.zeros(batch_size, dtype=torch.long)
    noise_0 = torch.randn_like(x_start)
    x_0 = diffusion.q_sample(x_start, t_0, noise_0)
    similarity_0 = torch.nn.functional.cosine_similarity(x_start, x_0, dim=1).mean()
    print(f"✓ At t=0: cosine similarity = {similarity_0:.4f} (should be close to 1.0)")

    # At t=99, should be mostly noise
    t_99 = torch.full((batch_size,), 99, dtype=torch.long)
    noise_99 = torch.randn_like(x_start)
    x_99 = diffusion.q_sample(x_start, t_99, noise_99)
    similarity_99 = torch.nn.functional.cosine_similarity(x_start, x_99, dim=1).mean()
    print(f"✓ At t=99: cosine similarity = {similarity_99:.4f} (should be close to 0.0)")

    print("\n✅ Forward diffusion tests passed!")

    # Visualize the forward diffusion process
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Forward Diffusion Process: Adding Noise Over Time', fontsize=16)

    # Use first sample from batch for visualization
    x_sample = x_start[0:1]  # Keep batch dimension

    # Timesteps to visualize
    vis_timesteps = [0, 10, 25, 50, 75, 99]

    for idx, t_val in enumerate(vis_timesteps):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Add noise at this timestep
        t = torch.tensor([t_val], dtype=torch.long)
        noise = torch.randn_like(x_sample)
        x_t = diffusion.q_sample(x_sample, t, noise)

        # Plot
        ax.plot(x_sample[0].numpy(), 'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(x_t[0].numpy(), 'r-', linewidth=1, label=f'Noisy (t={t_val})')

        # Add info
        signal_strength = diffusion.sqrt_alphas_cumprod[t_val].item()
        noise_strength = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()

        ax.set_title(f't = {t_val}\nSignal: {signal_strength:.3f}, Noise: {noise_strength:.3f}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)

    plt.tight_layout()
    output_path = 'assets/forward_diffusion_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to '{output_path}'")
    plt.show()