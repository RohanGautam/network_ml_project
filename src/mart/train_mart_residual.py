"""Train MART_ID to predict EqMotion's PLAYER residuals (boosting-style).

Loads the cache produced by cache_eqmotion_residuals.py:
    - past, target in MART's z-scored canonical frame
    - eqm_pred_ft: EqMotion ensemble mean in feet, canonical-ordered

Training:
    base_in_mart_z = (eqm_pred_ft - mart_mu) / mart_sigma   (residuals scale by sigma)
    residual_target_z = target - base_in_mart_z              (the residual to learn)
    MART forward(past) -> [B, N, K, T_f, 2] in mart-z; min_ade loss against
    residual_target_z, PLAYERS ONLY (canonical indices 0..9). Ball gradient is
    zeroed because the ball residual is irreducibly multimodal (we already
    proved this in the head-selector sweep).

Validation reports per-entity MSE in feet of the *combined* prediction:
    combined_ft = eqm_pred_ft;   combined_ft[:, players] += MART_residual_ft
This is the metric we actually care about — what Kaggle would score.

Usage:
    python train_mart_residual.py \\
        --cache_dir cache/eqm_residual \\
        --mart_config configs/mart_nba_pt.yaml \\
        --out_ckpt checkpoints/mart_residual_v1.pt \\
        --epochs 150
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from box import Box
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())

from models.mart_id import MART_ID  # noqa: E402
from utils import load_config  # noqa: E402

# Canonical order is [TeamA(5), TeamB(5), Ball(1)] → ball at index 10.
PLAYER_IDX = slice(0, 10)
BALL_IDX = 10


if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def _x_rel(x_abs):
    r = torch.zeros_like(x_abs)
    r[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    r[:, :, 0] = r[:, :, 1]
    return r


def residual_loss(y_pred, y_target_residual, loss_kind='min_ade'):
    """Loss in z-scored space, computed only on players (canonical [0..9]).

    y_pred:             [B, N, K, T_f, 2]
    y_target_residual:  [B, N, T_f, 2]  (already in z-scored residual form)

    loss_kind:
      - 'min_ade':  MART's native — min over K of mean L2 distance over T_f.
                    Lets K heads specialize on different residual modes.
      - 'mean_mse': all K heads pushed toward the same target. Forces a
                    deterministic regression on the mean residual conditioned
                    on past. No multimodal head specialization.
    """
    p = y_pred[:, PLAYER_IDX]               # [B, 10, K, T_f, 2]
    t = y_target_residual[:, PLAYER_IDX]    # [B, 10, T_f, 2]
    t_exp = t.unsqueeze(2)                  # [B, 10, 1, T_f, 2]
    if loss_kind == 'min_ade':
        per_k = torch.norm(p - t_exp, dim=-1).mean(dim=3)  # [B, 10, K]
        return per_k.min(dim=2).values.mean()
    if loss_kind == 'mean_mse':
        return ((p - t_exp) ** 2).mean()
    raise ValueError(f'unknown loss_kind {loss_kind}')


@torch.no_grad()
def evaluate(model, loader, mart_mu, mart_sigma, device):
    """Per-entity MSE in feet of (EqMotion + MART_player_residual) vs GT."""
    model.eval()
    # SSE accumulators per entity group
    sse = {'ball': 0.0, 'players': 0.0}
    cnt = {'ball': 0, 'players': 0}
    for past, target_z, eqm_pred_ft, agent_ids in loader:
        past, target_z, eqm_pred_ft, agent_ids = (
            t.to(device) for t in (past, target_z, eqm_pred_ft, agent_ids)
        )
        x_abs = past
        x_rel = _x_rel(x_abs)
        y_pred = model(x_abs, x_rel, agent_ids)  # [B, N, K, T_f, 2] residual in z
        mean_residual_z = y_pred.mean(dim=2)      # [B, N, T_f, 2]
        # Combine in feet:
        target_ft = target_z * mart_sigma + mart_mu
        combined_ft = eqm_pred_ft.clone()
        # Only update player slots — ball stays at EqMotion base.
        combined_ft[:, PLAYER_IDX] = (
            eqm_pred_ft[:, PLAYER_IDX]
            + mean_residual_z[:, PLAYER_IDX] * mart_sigma
        )
        sq = (combined_ft - target_ft) ** 2
        sse['ball'] += sq[:, BALL_IDX].sum().item()
        cnt['ball'] += sq[:, BALL_IDX].numel()
        sse['players'] += sq[:, PLAYER_IDX].sum().item()
        cnt['players'] += sq[:, PLAYER_IDX].numel()
    ball = sse['ball'] / max(cnt['ball'], 1)
    players = sse['players'] / max(cnt['players'], 1)
    return {
        'total11': (10 * players + ball) / 11,
        'ball': ball,
        'players': players,
    }


@torch.no_grad()
def evaluate_baseline_eqm(loader, mart_mu, mart_sigma, device):
    """Per-entity MSE of EqMotion alone (no residual) — the baseline to beat."""
    sse = {'ball': 0.0, 'players': 0.0}
    cnt = {'ball': 0, 'players': 0}
    for past, target_z, eqm_pred_ft, agent_ids in loader:
        target_ft = target_z.to(device) * mart_sigma + mart_mu
        eqm_pred_ft = eqm_pred_ft.to(device)
        sq = (eqm_pred_ft - target_ft) ** 2
        sse['ball'] += sq[:, BALL_IDX].sum().item()
        cnt['ball'] += sq[:, BALL_IDX].numel()
        sse['players'] += sq[:, PLAYER_IDX].sum().item()
        cnt['players'] += sq[:, PLAYER_IDX].numel()
    ball = sse['ball'] / max(cnt['ball'], 1)
    players = sse['players'] / max(cnt['players'], 1)
    return {
        'total11': (10 * players + ball) / 11,
        'ball': ball,
        'players': players,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--mart_config', required=True,
                   help='YAML config (only model hparams matter, not training).')
    p.add_argument('--out_ckpt', required=True)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-12)
    p.add_argument('--clip_grad', type=float, default=None)
    p.add_argument('--loss_kind', default='min_ade',
                   choices=['min_ade', 'mean_mse'],
                   help='min_ade: multimodal heads (collapses at mean-of-K). '
                        'mean_mse: deterministic regression on mean residual.')
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] device:', device)

    cd = Path(args.cache_dir)
    tr = torch.load(cd / 'train.pt', weights_only=False)
    vl = torch.load(cd / 'val.pt', weights_only=False)
    mart_mu = tr['mart_mu'].to(device)
    mart_sigma = tr['mart_sigma'].to(device)
    print(f'[INFO] train: {tr["past"].shape[0]} val: {vl["past"].shape[0]}')

    # ---- Build MART_ID from config; train from scratch on residuals ----
    opts = Box(load_config(args.mart_config))
    # We don't use hoops for residual training (pure 11-agent setup).
    opts.use_hoops = False
    n_entity_types = 3
    print(f'[INFO] MART config: model_dim={opts.model_dim}, sample_k={opts.sample_k}, '
          f'past={opts.past_length}, future={opts.future_length}')
    model = MART_ID(opts, num_entity_types=n_entity_types).to(device)
    print(f'[INFO] MART params: {sum(p.numel() for p in model.parameters())}')

    tr_ds = TensorDataset(tr['past'], tr['target'], tr['eqm_pred_ft'], tr['agent_ids'])
    vl_ds = TensorDataset(vl['past'], vl['target'], vl['eqm_pred_ft'], vl['agent_ids'])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=2, pin_memory=torch.cuda.is_available())
    vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, pin_memory=torch.cuda.is_available())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- Baselines ----
    base_val = evaluate_baseline_eqm(vl_loader, mart_mu, mart_sigma, device)
    print(f'[INFO] EqMotion-alone baseline val: '
          f'total11={base_val["total11"]:.4f} ball={base_val["ball"]:.4f} '
          f'players={base_val["players"]:.4f}')

    best_metric = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0; n = 0
        for past, target_z, eqm_pred_ft, agent_ids in tr_loader:
            past, target_z, eqm_pred_ft, agent_ids = (
                t.to(device) for t in (past, target_z, eqm_pred_ft, agent_ids)
            )
            # Residuals in MART's z-scored space (subtract base from target).
            # base_z = (eqm_pred_ft - mart_mu) / mart_sigma   (per-axis)
            # residual_z = target_z - base_z
            base_z = (eqm_pred_ft - mart_mu) / mart_sigma
            residual_target_z = target_z - base_z

            x_rel = _x_rel(past)
            y_pred = model(past, x_rel, agent_ids)  # [B, N, K, T_f, 2]
            loss = residual_loss(y_pred, residual_target_z, args.loss_kind)

            opt.zero_grad()
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            loss_sum += loss.item() * past.shape[0]
            n += past.shape[0]
        sched.step()

        val = evaluate(model, vl_loader, mart_mu, mart_sigma, device)
        # Track best by total11 (the Kaggle-aligned metric).
        improved = val['total11'] < best_metric
        if improved:
            best_metric = val['total11']
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        flag = '*' if improved else ' '
        print(f'  epoch {epoch:3d} | tr_loss {loss_sum/n:.5f} | '
              f'val total11 {val["total11"]:.4f} | ball {val["ball"]:.4f} '
              f'| players {val["players"]:.4f} {flag}')

    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': best_state,
        'opts': dict(opts),
        'mart_mu': mart_mu.cpu(),
        'mart_sigma': mart_sigma.cpu(),
        'best_val_total11': best_metric,
        'baseline_val': base_val,
        'hparams': vars(args),
    }, out)
    print(f'[INFO] best val total11: {best_metric:.4f} '
          f'(baseline {base_val["total11"]:.4f})')
    print(f'[INFO] saved residual MART to {out}')


if __name__ == '__main__':
    main()
