import torch
from torch import Tensor

# All functions expect inputs in the *original coordinate space* (i.e. denormalized).
# For this dataset: feet, court centered at origin (x ∈ ~[-47.5, 47.5], y ∈ ~[-25, 25]).
# Normalization is anisotropic: σ_x ≈ 29.7 ft, σ_y ≈ 11.5 ft — denorm via pred * sigma + mu
# restores true Euclidean distances. Passing normalized tensors gives unitless results
# that are not comparable across runs or with published numbers.


def compute_ade(pred: Tensor, target: Tensor) -> Tensor:
    """
    Average Displacement Error: mean L2 distance over all timesteps and agents.

    Args:
        pred:   [T, B*N, 2]  — denormalized, in feet
        target: [T, B*N, 2]  — denormalized, in feet
    Returns:
        scalar ADE in feet
    """
    return torch.norm(pred - target, dim=-1).mean()


def compute_mse(pred: Tensor, target: Tensor) -> Tensor:
    """
    MSE averaged per timestep — matches the Kaggle leaderboard metric.

    Args:
        pred:   [T, B*N, 2]  — denormalized, in feet
        target: [T, B*N, 2]  — denormalized, in feet
    Returns:
        scalar MSE in feet^2
    """
    return ((pred - target) ** 2).sum(dim=-1).mean()


def compute_fde(pred: Tensor, target: Tensor) -> Tensor:
    """
    Final Displacement Error: mean L2 distance at the last predicted timestep.

    Args:
        pred:   [T, B*N, 2]  — denormalized, in feet
        target: [T, B*N, 2]  — denormalized, in feet
    Returns:
        scalar FDE in feet
    """
    return torch.norm(pred[-1] - target[-1], dim=-1).mean()


def compute_min_ade(preds: Tensor, target: Tensor) -> Tensor:
    """
    minADE: minimum ADE over K stochastic predictions.

    Args:
        preds:  [K, T, B*N, 2]  — denormalized, in feet
        target: [T, B*N, 2]     — denormalized, in feet
    Returns:
        scalar minADE in feet
    """
    l2 = torch.norm(preds - target.unsqueeze(0), dim=-1)  # [K, T, B*N]
    ade_per_k = l2.mean(dim=1)  # [K, B*N]
    return ade_per_k.min(dim=0).values.mean()


def compute_min_fde(preds: Tensor, target: Tensor) -> Tensor:
    """
    minFDE: minimum FDE over K stochastic predictions.

    Args:
        preds:  [K, T, B*N, 2]  — denormalized, in feet
        target: [T, B*N, 2]     — denormalized, in feet
    Returns:
        scalar minFDE in feet
    """
    l2 = torch.norm(preds[:, -1] - target[-1].unsqueeze(0), dim=-1)  # [K, B*N]
    return l2.min(dim=0).values.mean()
