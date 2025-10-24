# 1D Diffusion Model - Learn by Building

This is a **minimal skeleton** for implementing a 1D diffusion model from scratch. The basic file structure is provided, but the core implementations are left empty for you to fill in as a learning exercise.

## Purpose

This repository is designed for those who want to learn diffusion models by implementing them step-by-step. By filling in the empty functions and classes yourself, you'll gain a deeper understanding of how diffusion models work.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `config.py` - Configuration parameters and hyperparameters
- `model.py` - Neural network model architecture
- `diffusion.py` - Core diffusion process (forward and reverse)
- `schedules.py` - Noise scheduling strategies
- `data.py` - Data loading and preprocessing
- `train.py` - Training loop implementation
- `sampling.py` - Sampling/generation from trained model
- `utils.py` - Helper functions and utilities
- `main.py` - Entry point for running the project

## Getting Started

1. Start by understanding the diffusion process theory
2. Implement the noise schedules in `schedules.py`
3. Build the forward diffusion process in `diffusion.py`
4. Create your model architecture in `model.py`
5. Implement the training loop in `train.py`
6. Add sampling functionality in `sampling.py`
7. Run and experiment with `main.py`

## Learning Resources

- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Excellent blog post by Lilian Weng
- DDPM Paper: "Denoising Diffusion Probabilistic Models" by Ho et al.

Fill in the implementations at your own pace and experiment with different approaches!
