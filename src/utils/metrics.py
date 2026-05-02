import torch
from torch import Tensor


def compute_ade(pred: Tensor, target: Tensor) -> Tensor:
    """
    Average Displacement Error: mean L2 distance over all timesteps and agents.

    Args:
        pred:   [T, B*N, 2]
        target: [T, B*N, 2]
    Returns:
        scalar
    """
    return torch.norm(pred - target, dim=-1).mean()


def compute_fde(pred: Tensor, target: Tensor) -> Tensor:
    """
    Final Displacement Error: mean L2 distance at the last predicted timestep.

    Args:
        pred:   [T, B*N, 2]
        target: [T, B*N, 2]
    Returns:
        scalar
    """
    return torch.norm(pred[-1] - target[-1], dim=-1).mean()


def compute_min_ade(preds: Tensor, target: Tensor) -> Tensor:
    """
    minADE: minimum ADE over K stochastic predictions.

    Args:
        preds:  [K, T, B*N, 2]
        target: [T, B*N, 2]
    Returns:
        scalar
    """
    l2 = torch.norm(preds - target.unsqueeze(0), dim=-1)  # [K, T, B*N]
    ade_per_k = l2.mean(dim=1)  # [K, B*N]
    return ade_per_k.min(dim=0).values.mean()


def compute_min_fde(preds: Tensor, target: Tensor) -> Tensor:
    """
    minFDE: minimum FDE over K stochastic predictions.

    Args:
        preds:  [K, T, B*N, 2]
        target: [T, B*N, 2]
    Returns:
        scalar
    """
    l2 = torch.norm(preds[:, -1] - target[-1].unsqueeze(0), dim=-1)  # [K, B*N]
    return l2.min(dim=0).values.mean()
