"""Training loop utilities for the 1-D diffusion model."""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from diffusion import Diffusion1D


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
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

    def train_epoch(self) -> Dict[str, float]:
        """Run a single training epoch and return aggregate metrics."""
        total_loss = 0.0
        num_batches = 0

        # Set model to training mode
        self.diffusion.network.train()

        for batch_idx, (clean_data, _) in enumerate(self.dataloader):
            # clean_data: [batch, seq_length]
            batch_size = clean_data.shape[0]

            # Sample random timesteps for each sequence in the batch
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)  # [batch]

            # Sample noise to add to the clean data
            noise = torch.randn_like(clean_data)  # [batch, seq_length]

            # Add noise according to the diffusion schedule (forward process)
            noisy_data = self.diffusion.q_sample(clean_data, t, noise)  # [batch, seq_length]

            # Predict the noise residuals using the model
            predicted_noise = self.diffusion.predict_noise(noisy_data, t)  # [batch, seq_length]

            # Compute loss between the true and predicted noise
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
        
        # Return average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}


    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run evaluation without gradient updates."""
        total_loss = 0.0
        num_batches = 0

        # Set model to evaluation mode
        self.diffusion.network.eval()

        # Disable gradient computation
        with torch.no_grad():
            for clean_data, _ in dataloader:
                batch_size = clean_data.shape[0]
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)  # [batch]
                noise = torch.randn_like(clean_data)
                noisy_data = self.diffusion.q_sample(clean_data, t, noise)
                predicted_noise = self.diffusion.predict_noise(noisy_data, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                total_loss += loss.item()
                num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"eval_loss": avg_loss}
                

    def save_checkpoint(self, path: str) -> None:
        """Persist model and optimizer state to disk."""
        checkpoint = {
            "model_state_dict": self.diffusion.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
