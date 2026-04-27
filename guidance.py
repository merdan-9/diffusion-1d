"""Model-based guidance modules for generated 1-D trajectories."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SineGuidanceConfig:
    """Configuration for sine-wave ODE guidance during sampling."""

    enabled: bool = False
    scale: float = 0.01
    start_fraction: float = 0.5
    amplitude_limit: float = 1.25
    amplitude_weight: float = 0.1
    min_frequency: float = 1.0
    max_frequency: float = 5.0
    num_steps: int = 1


class SineWaveGuidance:
    """Applies a small gradient correction toward sine-consistent sequences.

    The clean data is generated as sin(omega * t + phase) on t in [0, 2*pi].
    For a discrete sequence, the second difference approximately satisfies:

        x[i + 1] - 2*x[i] + x[i - 1] + (omega * dt)^2 * x[i] = 0

    The guidance estimates the discrete lambda=(omega*dt)^2 per sample and
    takes gradient steps that reduce this residual and amplitude violations.
    """

    def __init__(self, config: SineGuidanceConfig) -> None:
        self.config = config

    def should_apply(self, timestep: int, total_timesteps: int) -> bool:
        """Return whether guidance should run at this reverse-diffusion step."""
        if not self.config.enabled:
            return False

        start_timestep = int(total_timesteps * self.config.start_fraction)
        return timestep < start_timestep

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return a corrected copy of x using sine ODE residual gradients."""
        guided = x.detach()

        for _ in range(self.config.num_steps):
            guided = self._apply_once(guided)

        return guided

    def residual_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sine-consistency loss for a batch of sequences."""
        if x.shape[-1] < 3:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        x_mid = x[:, 1:-1]
        curvature = x[:, 2:] - 2.0 * x_mid + x[:, :-2]
        lambda_hat = self._estimate_discrete_lambda(x_mid, curvature)

        ode_residual = curvature + lambda_hat * x_mid
        ode_loss = (ode_residual ** 2).mean()

        amplitude_violation = torch.relu(x.abs() - self.config.amplitude_limit)
        amplitude_loss = (amplitude_violation ** 2).mean()

        return ode_loss + self.config.amplitude_weight * amplitude_loss

    def _apply_once(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_guided = x.detach().requires_grad_(True)
            loss = self.residual_loss(x_guided)
            grad = torch.autograd.grad(loss, x_guided)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            corrected = x_guided - self.config.scale * grad

        return corrected.detach()

    def _estimate_discrete_lambda(
        self,
        x_mid: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate lambda in curvature + lambda*x = 0 by least squares."""
        seq_length = x_mid.shape[-1] + 2
        dt = 2.0 * torch.pi / max(seq_length - 1, 1)

        min_lambda = (self.config.min_frequency * dt) ** 2
        max_lambda = (self.config.max_frequency * dt) ** 2

        numerator = -(curvature * x_mid).sum(dim=1, keepdim=True)
        denominator = (x_mid ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6)
        lambda_hat = numerator / denominator

        return lambda_hat.clamp(min=min_lambda, max=max_lambda)


def build_sine_guidance(config: Any) -> SineWaveGuidance:
    """Build sine guidance from a config object with matching attributes."""
    return SineWaveGuidance(
        SineGuidanceConfig(
            enabled=config.enable_sine_guidance,
            scale=config.sine_guidance_scale,
            start_fraction=config.sine_guidance_start_fraction,
            amplitude_limit=config.sine_guidance_amplitude_limit,
            amplitude_weight=config.sine_guidance_amplitude_weight,
            min_frequency=config.sine_guidance_min_frequency,
            max_frequency=config.sine_guidance_max_frequency,
            num_steps=config.sine_guidance_num_steps,
        )
    )
