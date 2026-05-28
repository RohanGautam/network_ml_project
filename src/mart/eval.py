"""Evaluate a trained MART / MART_ID checkpoint on the val split.

Unlike main_nba_pt.py's training-time validation (which reports min-of-K minADE
in *normalized* units), this script:
    - denormalizes predictions and targets back to feet using the checkpoint's
      (mu, sigma)
    - reports the full metric suite from network_ml_project's metrics module
      (ADE / FDE / MSE plus their min-of-K oracle versions)

Two prediction views are reported, mirroring submit_nba_pt.py's reduction:
    - mean-of-K: collapse K decoder heads by averaging (the Kaggle submission)
    - oracle min-of-K: best hypothesis per agent-trajectory

Example:
    python eval.py \\
        --checkpoint checkpoints/mart_pt_run1.ckpt \\
        --split_path ../network_ml_project/splits/fold0.json \\
        --gpu 0
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
from box import Box
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

# Load network_ml_project's metrics module by file path. We can't use a plain
# `from utils.metrics import ...` here because MART has its own utils.py module
# on sys.path, and per PEP 420 a regular module shadows a namespace package of
# the same name — so `import utils` resolves to MART/utils.py and then fails
# on `.metrics`. Loading by absolute file path bypasses the collision and
# leaves both modules independently usable.
HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE.parent / 'network_ml_project' / 'src' / 'utils' / 'metrics.py'
if not METRICS_PATH.exists():
    raise RuntimeError(
        f'Expected metrics.py at {METRICS_PATH} — adjust the path if your '
        f'network_ml_project layout differs.'
    )
_spec = importlib.util.spec_from_file_location('_nml_metrics', METRICS_PATH)
_nml_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nml_metrics)
compute_ade = _nml_metrics.compute_ade
compute_fde = _nml_metrics.compute_fde
compute_mse = _nml_metrics.compute_mse
compute_min_ade = _nml_metrics.compute_min_ade
compute_min_fde = _nml_metrics.compute_min_fde
compute_min_mse = _nml_metrics.compute_min_mse

from models.mart import MART   # noqa: E402
from models.mart_id import MART_ID   # noqa: E402
from loaders.dataloader_nba_pt import (   # noqa: E402
    MARTNBAPTDataset,
    WindowSampler,
    load_split_files,
)
from loaders.dataloader_nba_pt_hoops import (   # noqa: E402
    MARTNBAPTDataset as MARTNBAPTDatasetHoops,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID as HOOPS_ID,
)


if not torch.cuda.is_available():
    # Same CPU-compat patch as main_nba_pt.py / submit_nba_pt.py.
    print('CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat')
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate a MART checkpoint on val')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to a .ckpt produced by main_nba_pt.py')
    p.add_argument('--split_path', type=str,
                   default='../network_ml_project/splits/fold0.json',
                   help='Same split manifest used during training')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--seed', type=int, default=1,
                   help='Only affects WindowSampler start offsets')
    return p.parse_args()


def _x_rel_from_x_abs(x_abs):
    """Velocity tensor matching training/submission convention."""
    x_rel = torch.zeros_like(x_abs)
    x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    x_rel[:, :, 0] = x_rel[:, :, 1]
    return x_rel


def _model_forward(model, x_abs, x_rel, agent_ids):
    if isinstance(model, MART_ID):
        return model(x_abs, x_rel, agent_ids)
    return model(x_abs, x_rel)


def build_model(opts, device):
    use_id = bool(opts.get('use_entity_embed', False))
    if use_id:
        # If the checkpoint was trained with --use_hoops, the entity embedding
        # has an extra row for the basket class (id=3); rebuild with the same
        # size or load_state_dict will reject the shape mismatch.
        n_entity_types = (HOOPS_ID + 1) if opts.get('use_hoops', False) else 3
        print(
            f'[INFO] rebuilding MART_ID with embed_dim={opts.embed_dim}, '
            f'num_entity_types={n_entity_types}'
        )
        model = MART_ID(opts, num_entity_types=n_entity_types).to(device)
    else:
        print('[INFO] rebuilding stock MART (no entity embedding)')
        model = MART(opts).to(device)
    return model


def main():
    args = parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')

    # ---- Load checkpoint ----
    print(f'[INFO] loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    opts = Box(ckpt['opts'])
    mu = ckpt['mu'].float()         # [2]
    sigma = ckpt['sigma'].float()   # [2]
    use_hoops = bool(opts.get('use_hoops', False))
    print(
        f'[INFO] cfg: past={opts.past_length}, future={opts.future_length}, '
        f'sample_k={opts.sample_k}, '
        f'use_entity_embed={opts.get("use_entity_embed", False)}, '
        f'use_hoops={use_hoops}'
    )
    print(f'[INFO] norm stats: mu={mu.tolist()}, sigma={sigma.tolist()}')

    # ---- Build model and restore weights ----
    model = build_model(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    # ---- Val loader (re-use the checkpoint's mu/sigma) ----
    _, val_files = load_split_files(args.split_path)
    DatasetCls = MARTNBAPTDatasetHoops if use_hoops else MARTNBAPTDataset
    val_set = DatasetCls(
        val_files, mu, sigma, opts.past_length, opts.future_length,
    )
    val_sampler = WindowSampler(
        opts.batch_size, val_set.max_start, seed=args.seed, shuffle=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'[INFO] val sequences: {len(val_set)}')

    # ---- Inference, collected in metrics-module shape ----
    # Targets accumulate as [T_f, total, 2]; preds as [K, T_f, total, 2], in feet.
    mu_dev = mu.to(device)
    sigma_dev = sigma.to(device)
    pred_chunks = []
    target_chunks = []

    with torch.no_grad():
        for x_abs, y, agent_ids in val_loader:
            x_abs = x_abs.to(device)
            y = y.to(device)
            agent_ids = agent_ids.to(device)

            x_rel = _x_rel_from_x_abs(x_abs)
            y_pred = _model_forward(model, x_abs, x_rel, agent_ids)
            # y_pred: [B, N, K, T_f, 2]  (normalized)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            # Denormalize back to feet.
            y_pred_ft = y_pred * sigma_dev + mu_dev      # [B, N, K, T_f, 2]
            y_ft = y * sigma_dev + mu_dev                # [B, N, T_f, 2]

            # Drop hoop landmarks before reshaping: the Kaggle metric is over
            # the 11 real entities only, and hoops are trivially perfect so
            # they would deflate every reported number.
            if use_hoops:
                y_pred_ft = y_pred_ft[:, :HOOPS_N_REAL_AGENTS]
                y_ft = y_ft[:, :HOOPS_N_REAL_AGENTS]

            # Reshape to the metrics module's [(K,) T, B*N, 2] convention.
            B, N, K, T_f, _ = y_pred_ft.shape
            y_pred_ft = (
                y_pred_ft.permute(2, 3, 0, 1, 4).reshape(K, T_f, B * N, 2)
            )
            y_ft = y_ft.permute(2, 0, 1, 3).reshape(T_f, B * N, 2)

            pred_chunks.append(y_pred_ft.cpu())
            target_chunks.append(y_ft.cpu())

    preds_all = torch.cat(pred_chunks, dim=2)      # [K, T_f, total, 2]
    target_all = torch.cat(target_chunks, dim=1)   # [T_f, total, 2]
    K, T_f, total, _ = preds_all.shape
    print(
        f'[INFO] evaluating {total} agent-trajectories | K={K} heads | '
        f'T_f={T_f} steps'
    )

    # Mean-of-K reduction is what submit_nba_pt.py writes by default
    # (optimal point estimate under MSE for equally-weighted hypotheses).
    preds_mean = preds_all.mean(dim=0)             # [T_f, total, 2]

    metrics = {
        # Single-mode metrics on the mean-of-K trajectory (= submission).
        'submission/ADE (ft)':  compute_ade(preds_mean, target_all).item(),
        'submission/FDE (ft)':  compute_fde(preds_mean, target_all).item(),
        'submission/MSE (ft^2)': compute_mse(preds_mean, target_all).item(),
        # Oracle best-of-K metrics on the full K hypotheses.
        f'oracle/minADE@K={K} (ft)':  compute_min_ade(preds_all, target_all).item(),
        f'oracle/minFDE@K={K} (ft)':  compute_min_fde(preds_all, target_all).item(),
        f'oracle/minMSE@K={K} (ft^2)': compute_min_mse(preds_all, target_all).item(),
    }

    print('\n=== Val metrics (denormalized, feet) ===')
    width = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f'  {k:<{width}} : {v:.4f}')


if __name__ == '__main__':
    main()
