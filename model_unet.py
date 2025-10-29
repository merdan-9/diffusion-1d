"""UNet modules for predicting diffusion noise on 1-D trajectories."""

import torch
from torch import nn

from model import SinusoidalTimeEmbedding


class ConvBlock(nn.Module):
    """Single convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels:int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, seq_length]
        Returns:
            out: [batch, out_channels, seq_length]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    """Encoder block with convolution and downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, in_channels, seq_length]
        Returns:
            skip: [batch, out_channels, seq_length] (for skip connection)
            out: [batch, out_channels, seq_length // 2] (after downsampling)
        """
        x = self.conv1(x)         # [batch, out_channels, seq_length]
        x = self.conv2(x)         # [batch, out_channels, seq_length]
        skip = x                  # Save for skip connection
        x = self.downsample(x)    # [batch, out_channels, seq_length // 2]
        return skip, x

class UpBlock(nn.Module):
    """Decoder block with upsampling and convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, seq_length] (from previous layer)
            skip: [batch, skip_channels, seq_length * 2] (from encoder)
        Returns:
            out: [batch, out_channels, seq_length * 2]
        """
        x = self.upsample(x)             # [batch, in_channels, seq_length * 2]
        x = torch.cat([x, skip], dim=1)  # [batch, in_channels + skip_channels, seq_length * 2]
        x = self.conv1(x)                # [batch, out_channels, seq_length * 2]
        x = self.conv2(x)                # [batch, out_channels, seq_length * 2]
        return x
    

class UNet1D(nn.Module):
    """1-D UNet architecture for diffusion noise prediction.

    Interface matches NoisePredictor (MLP) for easy swapping:
    - Input: [batch, seq_length] and raw timesteps [batch]
    - Output: [batch, seq_length]
    """

    def __init__(self, seq_length: int, hidden_dim: int, time_dim: int):
        """Initialize UNet with same signature as NoisePredictor (MLP).

        Args:
            seq_length: Length of input sequences (not used, kept for compatibility)
            hidden_dim: Base number of channels (like MLP's hidden_dim)
            time_dim: Dimension of time embeddings
        """
        super().__init__()

        # Time embedding module (same as MLP)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # Internal UNet uses channel dimension
        in_channels = 1  # Input has 1 channel internally
        base_channels = hidden_dim  # Use hidden_dim as base channel count

        # Channel progression: 1 → 32 → 32 → 64 → 128 (encoder)
        #                      128 → 64 → 32 → 32 → 1 (decoder)

        self.initial_conv = ConvBlock(in_channels, base_channels)

        # Encoder: progressively downsample and increase channels
        self.down1 = DownBlock(base_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)

        # Bottleneck: process at smallest spatial resolution
        self.bottleneck_conv1 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck_conv2 = ConvBlock(base_channels * 8, base_channels * 4)

        # Decoder: progressively upsample and decrease channels
        # Note: skip_channels must match corresponding encoder output
        self.up3 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)  # Uses skip3
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)      # Uses skip2
        self.up1 = UpBlock(base_channels, base_channels, base_channels)              # Uses skip1

        self.final_conv = nn.Conv1d(base_channels, in_channels, kernel_size=1)

        # Time embedding: injected at bottleneck (deepest point)
        self.time_proj = nn.Linear(time_dim, base_channels * 4)

    def forward(self, noisy_sequence: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict noise residuals (same interface as NoisePredictor MLP).

        Args:
            noisy_sequence: [batch, seq_length] - noisy input (same as MLP)
            timesteps: [batch] - diffusion timesteps as integers (same as MLP)
        Returns:
            [batch, seq_length] - predicted noise (same as MLP)
        """
        # Add channel dimension for UNet internal processing
        x = noisy_sequence.unsqueeze(1)  # [batch, seq_length] → [batch, 1, seq_length]

        # Create time embeddings internally (same as MLP does)
        t_emb = self.time_embedding(timesteps)  # [batch] → [batch, time_dim]

        # UNet processing with channel dimension
        x = self.initial_conv(x)

        # Encoder: save skip connections at each resolution
        skip1, x = self.down1(x)  # 64 length
        skip2, x = self.down2(x)  # 32 length
        skip3, x = self.down3(x)  # 16 length
        # x is now at 8 length (smallest)

        # Inject time information at bottleneck
        t_emb_proj = self.time_proj(t_emb)    # [batch, base_channels * 4]
        t_emb_proj = t_emb_proj[:, :, None]   # [batch, base_channels * 4, 1]
        x = x + t_emb_proj                    # Broadcast addition

        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)

        # Decoder: pair each upsampling with corresponding skip connection
        x = self.up3(x, skip3)  # 8 → 16 length, combine with skip3 (16 length)
        x = self.up2(x, skip2)  # 16 → 32 length, combine with skip2 (32 length)
        x = self.up1(x, skip1)  # 32 → 64 length, combine with skip1 (64 length)

        out = self.final_conv(x)  # [batch, 1, seq_length]

        # Remove channel dimension to match MLP output
        out = out.squeeze(1)  # [batch, 1, seq_length] → [batch, seq_length]

        return out


if __name__ == "__main__":
    print("="*60)
    print("Testing UNet1D Components")
    print("="*60)

    # Test 1: ConvBlock
    print("\n[Test 1] ConvBlock")
    print("-" * 40)
    conv_block = ConvBlock(in_channels=16, out_channels=32)
    x_test = torch.randn(4, 16, 64)  # [batch, channels, length]
    out = conv_block(x_test)
    print(f"✓ Input shape:  {x_test.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (4, 32, 64), "ConvBlock output shape incorrect!"
    print("✓ ConvBlock test passed!")

    # Test 2: DownBlock
    print("\n[Test 2] DownBlock")
    print("-" * 40)
    down_block = DownBlock(in_channels=32, out_channels=64)
    x_test = torch.randn(4, 32, 64)
    skip, out = down_block(x_test)
    print(f"✓ Input shape:  {x_test.shape}")
    print(f"✓ Skip shape:   {skip.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert skip.shape == (4, 64, 64), "DownBlock skip shape incorrect!"
    assert out.shape == (4, 64, 32), "DownBlock output shape incorrect!"
    print("✓ DownBlock test passed!")

    # Test 3: UpBlock
    print("\n[Test 3] UpBlock")
    print("-" * 40)
    up_block = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
    x_test = torch.randn(4, 128, 16)  # From deeper layer
    skip_test = torch.randn(4, 64, 32)  # From encoder
    out = up_block(x_test, skip_test)
    print(f"✓ Input shape:      {x_test.shape}")
    print(f"✓ Skip shape:       {skip_test.shape}")
    print(f"✓ Output shape:     {out.shape}")
    assert out.shape == (4, 64, 32), "UpBlock output shape incorrect!"
    print("✓ UpBlock test passed!")

    # Test 4: Complete UNet1D
    print("\n[Test 4] Complete UNet1D (MLP-compatible interface)")
    print("-" * 40)
    unet = UNet1D(seq_length=64, hidden_dim=32, time_dim=128)

    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"✓ Total parameters:     {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Test forward pass with MLP-compatible interface
    x_test = torch.randn(4, 64)  # [batch, seq_length] - Same as MLP!
    t_test = torch.randint(0, 100, (4,))  # [batch] - Raw timesteps, same as MLP!
    out = unet(x_test, t_test)
    print(f"✓ Input shape:    {x_test.shape}")
    print(f"✓ Timesteps shape: {t_test.shape}")
    print(f"✓ Output shape:   {out.shape}")
    assert out.shape == (4, 64), "UNet1D output shape incorrect!"
    print("✓ UNet1D test passed!")