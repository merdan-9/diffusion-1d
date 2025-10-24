"""Neural network modules for predicting diffusion noise on 1-D trajectories."""

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Transforms scalar diffusion timesteps into learned embeddings."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        """Initialize embedding parameters."""
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Produce embeddings for the provided diffusion timesteps."""
        # timesteps: [batch]
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period)) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)

        # Compute the sinusoidal embeddings
        # timesteps: [batch] -> [batch, 1]
        # freqs: [half_dim] -> [1, half_dim]
        # args: [batch, half_dim]
        args = timesteps[:, None].float() * freqs[None, :]

        # Combine cosines and sines
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # [batch, dim]
        return embedding


class NoisePredictor(nn.Module):
    """Predicts noise residuals from noisy trajectories and time embeddings."""

    def __init__(self, seq_length: int, hidden_dim: int, time_dim: int) -> None:
        """Construct the temporal network backbone."""
        super().__init__()
        # Time embedding module
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # Network layers
        self.input_proj = nn.Linear(seq_length + time_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, seq_length)
        self.act = nn.ReLU()

    def forward(self, noisy_sequence: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return predicted noise residuals."""
        # noisy_sequence: [batch, seq_length]
        # timesteps: [batch]

        # Get time embeddings
        t_emb = self.time_embedding(timesteps)

        # Concatenate time embeddings to the noisy sequence
        x = torch.cat([noisy_sequence, t_emb], dim=-1) # [batch, seq_length + time_dim]

        # Pass through the network
        x = self.act(self.input_proj(x))
        x = self.act(self.hidden(x))
        x = self.output_proj(x) # [batch, seq_length]
        return x

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    print("Testing NoisePredictor network...")

    # Create output folder
    os.makedirs("assets", exist_ok=True)

    # Test parameters
    batch_size = 4
    seq_length = 64
    hidden_dim = 128
    time_dim = 128

    # Test 1: Time embedding
    print("\n1. Testing SinusoidalTimeEmbedding...")
    time_emb = SinusoidalTimeEmbedding(dim=time_dim)

    timesteps = torch.tensor([0, 10, 25, 50, 75, 99])
    embeddings = time_emb(timesteps)

    print(f"✓ Input timesteps shape: {timesteps.shape}")
    print(f"✓ Output embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (len(timesteps), time_dim), "Time embedding output shape is incorrect."

    diff = torch.abs(embeddings[0] - embeddings[-1]).mean()
    print(f"✓ Mean absolute difference between first and last embedding: {diff:.6f}")

    # Test 2: Noise predictor network
    print("\n2. Testing NoisePredictor network...")

    model = NoisePredictor(seq_length=seq_length, hidden_dim=hidden_dim, time_dim=time_dim)

    # Create dummy inputs
    noisy_sequences = torch.randn(batch_size, seq_length)
    timesteps = torch.randint(0, 100, (batch_size,))

    # Forward pass
    predicted_noise = model(noisy_sequences, timesteps)

    print(f"✓ Input noisy sequences shape: {noisy_sequences.shape}")
    print(f"✓ Input timesteps shape: {timesteps.shape}")
    print(f"✓ Output predicted noise shape: {predicted_noise.shape}")
    assert predicted_noise.shape == (batch_size, seq_length), "Predicted noise output shape is incorrect."

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total trainable parameters in NoisePredictor: {total_params}")

    # Visualization 1: Time Embeddings Heatmap
    print("\n3. Visualizing time embeddings...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Generate embeddings for many timesteps
    all_timesteps = torch.arange(0, 100)
    all_embeddings = time_emb(all_timesteps)  # [100, time_dim]

    # Plot 1: Heatmap of embeddings
    ax = axes[0]
    im = ax.imshow(all_embeddings.T.detach().numpy(), aspect='auto', cmap='RdBu')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Time Embeddings Heatmap\n(Each column = one timestep)')
    plt.colorbar(im, ax=ax)

    # Plot 2: Sample embedding dimensions over time
    ax = axes[1]
    # Plot a few embedding dimensions
    for dim_idx in [0, 16, 32, 48, 64]:
        ax.plot(all_timesteps.numpy(),
                all_embeddings[:, dim_idx].detach().numpy(),
                label=f'Dim {dim_idx}',
                alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Embedding Value')
    ax.set_title('Sample Embedding Dimensions Over Time\n(Different frequencies)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/time_embeddings_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'assets/time_embeddings_visualization.png'")
    plt.show()

    # Visualization 2: Model predictions (untrained)
    print("\n4. Visualizing model predictions (untrained)...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Untrained Model Predictions', fontsize=16)

    # Create a sine wave as test input
    x = torch.linspace(0, 2 * torch.pi, seq_length)
    clean_signal = torch.sin(x)

    for idx, t_val in enumerate([0, 25, 50, 99]):
        ax = axes[idx // 2, idx % 2]

        # Add some noise
        noisy_signal = clean_signal + 0.5 * torch.randn_like(clean_signal)

        # Predict noise
        t = torch.tensor([t_val])
        predicted_noise = model(noisy_signal.unsqueeze(0), t)

        # Plot
        ax.plot(clean_signal.numpy(), 'b-', label='Clean signal', linewidth=2, alpha=0.7)
        ax.plot(noisy_signal.numpy(), 'gray', label='Noisy signal', alpha=0.5)
        ax.plot(predicted_noise[0].detach().numpy(), 'r--', label='Predicted noise', linewidth=2)

        ax.set_title(f'Timestep t={t_val}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/model_predictions_untrained.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'assets/model_predictions_untrained.png'")

    plt.show()

    print("\n✅ All model tests passed!")