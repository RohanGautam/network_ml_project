"""Per-entity MSE breakdown for a MART checkpoint.

Mirrors eval.py's full feet-units evaluation pipeline but splits the final
metric into ball vs players to answer: 'is MART's ball MSE in a different
regime than EqMotion's ~13.7?'

After canonical reorder + hoop-stripping the entity axis is exactly
    [TeamA(5), TeamB(5), Ball(1)]  -> index 10 is the ball, 0..9 are players.

Reports both:
  * mean-of-K (the Kaggle-relevant point estimate)
  * oracle min-of-K (best hypothesis per agent — tells us how multimodal
    the predictions are; if ball minMSE << ball MSE, K=20 is doing real work
    on the ball but the mean-collapse is wasting it)

Usage:
    python diagnose_per_entity.py --checkpoint checkpoints/mart_minade_s1.ckpt
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

# Re-use eval.py's metrics import trick — utils.py in MART shadows the project's
# utils package, so load the metrics module by file path.
HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE.parents[1] / 'src' / 'utils' / 'metrics.py'
_spec = importlib.util.spec_from_file_location('_nml_metrics', METRICS_PATH)
_nml_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nml_metrics)
compute_mse = _nml_metrics.compute_mse
compute_min_mse = _nml_metrics.compute_min_mse

from models.mart import MART  # noqa: E402
from models.mart_id import MART_ID  # noqa: E402
from loaders.dataloader_nba_pt import (  # noqa: E402
    MARTNBAPTDataset,
    WindowSampler,
    load_split_files,
)
from loaders.dataloader_nba_pt_hoops import (  # noqa: E402
    MARTNBAPTDataset as MARTNBAPTDatasetHoops,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID as HOOPS_ID,
)


if not torch.cuda.is_available():
    print('CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat')
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


# Canonical order is [TeamA(5), TeamB(5), Ball(1)], so:
BALL_IDX = 10           # last of the 11 real entities
PLAYER_IDXS = list(range(10))  # indices 0..9


def _x_rel_from_x_abs(x_abs):
    x_rel = torch.zeros_like(x_abs)
    x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    x_rel[:, :, 0] = x_rel[:, :, 1]
    return x_rel


def _model_forward(model, x_abs, x_rel, agent_ids):
    if isinstance(model, MART_ID):
        return model(x_abs, x_rel, agent_ids)
    return model(x_abs, x_rel)


def _build_model(opts, device):
    use_id = bool(opts.get('use_entity_embed', False))
    if use_id:
        n_entity_types = (HOOPS_ID + 1) if opts.get('use_hoops', False) else 3
        return MART_ID(opts, num_entity_types=n_entity_types).to(device)
    return MART(opts).to(device)


def evaluate(checkpoint_path, split_path, batch_size, num_workers, device):
    print(f'\n========== {Path(checkpoint_path).name} ==========')
    # weights_only=False is needed because the checkpoint stores a Box config
    # object (python-box) which the safe unpickler refuses by default. The
    # checkpoints are produced by our own training code, so this is safe.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    mu = ckpt['mu'].float()
    sigma = ckpt['sigma'].float()
    use_hoops = bool(opts.get('use_hoops', False))
    print(
        f'  cfg: K={opts.sample_k}, past={opts.past_length}, '
        f'future={opts.future_length}, use_hoops={use_hoops}, '
        f'use_entity_embed={opts.get("use_entity_embed", False)}'
    )

    model = _build_model(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    _, val_files = load_split_files(split_path)
    DatasetCls = MARTNBAPTDatasetHoops if use_hoops else MARTNBAPTDataset
    val_set = DatasetCls(
        val_files, mu, sigma, opts.past_length, opts.future_length,
    )
    val_sampler = WindowSampler(batch_size, val_set.max_start, seed=1, shuffle=False)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    mu_d = mu.to(device)
    sigma_d = sigma.to(device)
    # Accumulate sum-of-squared-errors and counts per (entity-group, K-reduction).
    sse = {('mean', 'ball'): 0.0, ('mean', 'players'): 0.0,
           ('mink', 'ball'): 0.0, ('mink', 'players'): 0.0}
    cnt = {k: 0 for k in sse}

    with torch.no_grad():
        for x_abs, y, agent_ids in val_loader:
            x_abs, y, agent_ids = (
                x_abs.to(device), y.to(device), agent_ids.to(device)
            )
            x_rel = _x_rel_from_x_abs(x_abs)
            y_pred = _model_forward(model, x_abs, x_rel, agent_ids)  # [B,N,K,T,2]
            if opts.pred_rel:
                cur = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur

            y_pred_ft = y_pred * sigma_d + mu_d
            y_ft = y * sigma_d + mu_d

            if use_hoops:
                y_pred_ft = y_pred_ft[:, :HOOPS_N_REAL_AGENTS]
                y_ft = y_ft[:, :HOOPS_N_REAL_AGENTS]
            # Now y_pred_ft [B, 11, K, T, 2], y_ft [B, 11, T, 2]

            # Mean-of-K point estimate (the Kaggle submission), per entity.
            mean_pred = y_pred_ft.mean(dim=2)  # [B, 11, T, 2]
            sq = (mean_pred - y_ft) ** 2       # [B, 11, T, 2]

            # Oracle min-of-K MSE per entity: per-trajectory MSE-per-K then min.
            #   K view: [B, 11, K, T, 2] -> per-(B,N,K) MSE over (T, axes) -> min over K
            sq_k = (y_pred_ft - y_ft.unsqueeze(2)) ** 2  # [B, 11, K, T, 2]
            mse_per_k = sq_k.mean(dim=(3, 4))            # [B, 11, K]
            mse_mink = mse_per_k.min(dim=2).values        # [B, 11]

            # Accumulate ball (idx 10) and players (idx 0..9) separately.
            ball_sq = sq[:, BALL_IDX]           # [B, T, 2]
            player_sq = sq[:, :BALL_IDX]        # [B, 10, T, 2]
            sse[('mean', 'ball')]    += ball_sq.sum().item();    cnt[('mean', 'ball')]    += ball_sq.numel()
            sse[('mean', 'players')] += player_sq.sum().item();  cnt[('mean', 'players')] += player_sq.numel()
            # For min-of-K, the "MSE" is already a mean over (T, axes), so each
            # entry counts once per (batch, entity). Accumulate sum and count
            # of these per-(B,N) MSEs and divide at the end.
            ball_mink = mse_mink[:, BALL_IDX]           # [B]
            player_mink = mse_mink[:, :BALL_IDX]        # [B, 10]
            sse[('mink', 'ball')]    += ball_mink.sum().item();    cnt[('mink', 'ball')]    += ball_mink.numel()
            sse[('mink', 'players')] += player_mink.sum().item();  cnt[('mink', 'players')] += player_mink.numel()

    res = {k: sse[k] / max(cnt[k], 1) for k in sse}
    res[('mean', 'total11')] = (
        (10 * res[('mean', 'players')] + res[('mean', 'ball')]) / 11
    )
    res[('mink', 'total11')] = (
        (10 * res[('mink', 'players')] + res[('mink', 'ball')]) / 11
    )
    print('  ----------------- val/mse_ft (denorm, feet²) -----------------')
    print(f'  {"reduction":12s} {"total11":>10s} {"ball":>10s} {"players":>10s}  ball/players')
    for red in ('mean', 'mink'):
        total = res[(red, 'total11')]
        ball = res[(red, 'ball')]
        play = res[(red, 'players')]
        ratio = ball / max(play, 1e-9)
        tag = 'mean-of-K' if red == 'mean' else f'min-of-K(={opts.sample_k})'
        print(f'  {tag:12s} {total:10.4f} {ball:10.4f} {play:10.4f}  {ratio:5.2f}x')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='One or more .ckpt paths from main_nba_pt.py')
    p.add_argument('--split_path', type=str,
                   default='../network_ml_project/splits/fold0.json')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')

    for ck in args.checkpoints:
        evaluate(ck, args.split_path, args.batch_size, args.num_workers, device)


if __name__ == '__main__':
    main()
