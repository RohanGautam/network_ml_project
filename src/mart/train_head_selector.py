"""Train a per-agent learned head selector for MART's K=20 outputs.

Loads cached MART predictions (from cache_mart_preds.py) and trains a small
per-agent MLP that maps:
    (past trajectory, past velocity, agent_id, K candidate predictions)
            ->  softmax weights over K  ->  weighted prediction
End-to-end MSE loss on the head-weighted prediction (in z-scored space).

The model never sees an extra MART forward — all K predictions are cached.
Val metrics report per-entity MSE in feet² (ball, players, total11), the same
honest metric used everywhere else in this project.

Usage:
    python train_head_selector.py \\
        --cache_dir ../../cache/mart_minade_s1 \\
        --out_ckpt checkpoints/head_selector_v1.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.getcwd())

# Canonical agent order is [TeamA(5), TeamB(5), Ball(1)] -> ball at index 10.
BALL_IDX = 10


def _vel(past):
    v = torch.zeros_like(past)
    v[:, :, 1:] = past[:, :, 1:] - past[:, :, :-1]
    v[:, :, 0] = v[:, :, 1]
    return v


class HeadSelector(nn.Module):
    """Per-agent MLP -> K-way softmax over MART's heads.

    Each agent gets its own past + velocity + id + K candidate trajectories,
    PLUS a flattened scene-context vector containing every agent's past pos
    and velocity. The scene context is critical for the ball: picking the
    right mode (pass-to-A vs pass-to-B vs drive) depends on where the
    receivers and defenders are, not just the ball's own history.
    """

    def __init__(self, past_len, fut_len, K, n_agent_types, n_agents=11,
                 hidden=128, dropout=0.1, scene_context=True):
        super().__init__()
        self.K = K
        self.n_agent_types = n_agent_types
        self.scene_context = scene_context
        scene_dim = (n_agents * past_len * 2 * 2) if scene_context else 0  # all agents' pos+vel
        in_dim = (
            past_len * 2     # this agent's past pos
            + past_len * 2   # this agent's past vel
            + n_agent_types  # this agent's id one-hot
            + K * fut_len * 2  # this agent's K candidate trajectories
            + scene_dim      # ALL agents' past pos+vel (the scene)
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, K),
        )

    def forward(self, past, agent_ids, k_preds):
        # past [B, N, T_p, 2], agent_ids [B, N], k_preds [B, N, K, T_f, 2]
        B, N = past.shape[:2]
        past_flat = past.reshape(B, N, -1)            # [B, N, T_p*2]
        vel_flat = _vel(past).reshape(B, N, -1)       # [B, N, T_p*2]
        ids_oh = F.one_hot(agent_ids.long(), self.n_agent_types).float()
        kp_flat = k_preds.reshape(B, N, -1)           # [B, N, K*T_f*2]
        parts = [past_flat, vel_flat, ids_oh, kp_flat]
        if self.scene_context:
            # Flatten every agent's pos+vel into one scene vector and broadcast
            # to every agent's input. Order matters but is canonical and fixed.
            scene = torch.cat([past_flat, vel_flat], dim=-1)  # [B, N, T_p*4]
            scene = scene.reshape(B, -1)                       # [B, N*T_p*4]
            scene = scene.unsqueeze(1).expand(B, N, -1)        # [B, N, N*T_p*4]
            parts.append(scene)
        x = torch.cat(parts, dim=-1)
        return self.net(x)  # [B, N, K]


def weighted_pred(logits, k_preds):
    """Softmax-weighted combination over K heads. -> [B, N, T_f, 2]."""
    w = F.softmax(logits, dim=-1)
    return (w.unsqueeze(-1).unsqueeze(-1) * k_preds).sum(dim=2)


def per_entity_mse_ft(pred_n, target_n, mu, sigma):
    """Per-entity MSE in feet². Inputs are in z-scored space."""
    p = pred_n * sigma + mu
    t = target_n * sigma + mu
    sq = (p - t) ** 2  # [B, N, T, 2]
    out = {}
    out['ball'] = sq[:, BALL_IDX].sum().item()
    out['ball_n'] = sq[:, BALL_IDX].numel()
    out['players'] = sq[:, :BALL_IDX].sum().item()
    out['players_n'] = sq[:, :BALL_IDX].numel()
    return out


def epoch_val_per_entity(model, loader, mu_d, sigma_d, device, baseline=False):
    """Accumulate per-entity MSE over the loader. baseline=True ignores the
    selector and uses raw mean-of-K (for the diagnostic comparison)."""
    model.eval()
    acc = {'ball': 0.0, 'ball_n': 0, 'players': 0.0, 'players_n': 0}
    with torch.no_grad():
        for past, target, k_preds, agent_ids in loader:
            past, target, k_preds, agent_ids = (
                t.to(device) for t in (past, target, k_preds, agent_ids)
            )
            if baseline:
                pred = k_preds.mean(dim=2)
            else:
                logits = model(past, agent_ids, k_preds)
                pred = weighted_pred(logits, k_preds)
            r = per_entity_mse_ft(pred, target, mu_d, sigma_d)
            for k in acc:
                acc[k] += r[k]
    ball = acc['ball'] / max(acc['ball_n'], 1)
    players = acc['players'] / max(acc['players_n'], 1)
    return {
        'total11': (10 * players + ball) / 11,
        'ball': ball,
        'players': players,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--out_ckpt', required=True)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--ball_loss_weight', type=float, default=1.0,
                   help='Per-element ball loss weight in training. 1.0 = plain '
                        'mean MSE (matches Kaggle); >1 prioritizes ball.')
    p.add_argument('--ball_only_loss', action='store_true',
                   help='Train selector with ball-only loss (zero gradient on '
                        'players). Use when the goal is ball-specialist '
                        'predictions for combining with another model\'s '
                        'player predictions.')
    p.add_argument('--monitor', default='ball',
                   choices=['total11', 'ball'],
                   help='Pick best epoch by this val metric.')
    p.add_argument('--no_scene_context', action='store_true',
                   help='Disable the all-agents-pos+vel scene-context feature.')
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] device:', device)

    cd = Path(args.cache_dir)
    tr = torch.load(cd / 'train.pt', weights_only=False)
    vl = torch.load(cd / 'val.pt', weights_only=False)
    K = tr['k_preds'].shape[2]
    past_len = tr['past'].shape[2]
    fut_len = tr['target'].shape[2]
    n_agent_types = int(tr['agent_ids'].max().item()) + 1
    mu_d = tr['mu'].to(device)
    sigma_d = tr['sigma'].to(device)
    print(f'[INFO] train:{tr["past"].shape[0]} val:{vl["past"].shape[0]} '
          f'K={K} past_len={past_len} fut_len={fut_len} '
          f'n_agent_types={n_agent_types}')

    tr_ds = TensorDataset(tr['past'], tr['target'], tr['k_preds'], tr['agent_ids'])
    vl_ds = TensorDataset(vl['past'], vl['target'], vl['k_preds'], vl['agent_ids'])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=2, pin_memory=torch.cuda.is_available())
    vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=2, pin_memory=torch.cuda.is_available())

    n_agents = tr['past'].shape[1]
    selector = HeadSelector(
        past_len, fut_len, K, n_agent_types, n_agents=n_agents,
        hidden=args.hidden, dropout=args.dropout,
        scene_context=not args.no_scene_context,
    ).to(device)
    print(f'[INFO] selector params: '
          f'{sum(p.numel() for p in selector.parameters())}')
    opt = torch.optim.AdamW(selector.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Baselines for comparison: mean-of-K (the current MART submission).
    baseline_val = epoch_val_per_entity(selector, vl_loader, mu_d, sigma_d,
                                        device, baseline=True)
    print('[INFO] BASELINE (mean-of-K) val: '
          f'total11={baseline_val["total11"]:.4f} '
          f'ball={baseline_val["ball"]:.4f} '
          f'players={baseline_val["players"]:.4f}')

    best_metric = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        selector.train()
        tr_loss_sum = 0.0
        tr_n = 0
        for past, target, k_preds, agent_ids in tr_loader:
            past, target, k_preds, agent_ids = (
                t.to(device) for t in (past, target, k_preds, agent_ids)
            )
            logits = selector(past, agent_ids, k_preds)
            pred = weighted_pred(logits, k_preds)
            sq = (pred - target) ** 2  # [B, N, T, 2] (z-scored)
            if args.ball_only_loss:
                # Ball-only: zero gradient on player predictions; selector
                # specializes entirely on picking the right head for the ball.
                loss = sq[:, BALL_IDX].mean()
            elif args.ball_loss_weight == 1.0:
                loss = sq.mean()
            else:
                # Per-element weighted mean: ball elements count `ball_loss_weight`x.
                B, N, T, _ = sq.shape
                w = torch.ones(N, device=device)
                w[BALL_IDX] = args.ball_loss_weight
                w_full = w.view(1, N, 1, 1).expand_as(sq)
                loss = (w_full * sq).sum() / w_full.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * past.shape[0]
            tr_n += past.shape[0]
        sched.step()

        val = epoch_val_per_entity(selector, vl_loader, mu_d, sigma_d, device)
        metric = val[args.monitor]
        improved = metric < best_metric
        if improved:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone()
                          for k, v in selector.state_dict().items()}
        flag = '*' if improved else ' '
        print(f'  epoch {epoch:3d} | tr_loss {tr_loss_sum/tr_n:.5f} | '
              f'val total11 {val["total11"]:.4f} | ball {val["ball"]:.4f} '
              f'| players {val["players"]:.4f} {flag}')

    print(f'\n[INFO] best monitor={args.monitor} val: {best_metric:.4f}')
    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'selector': best_state,
        'hparams': vars(args),
        'K': K,
        'past_len': past_len,
        'fut_len': fut_len,
        'n_agent_types': n_agent_types,
        'best_metric': best_metric,
        'baseline_mean_of_K': baseline_val,
    }, out)
    print(f'[INFO] saved selector to {out}')


if __name__ == '__main__':
    main()
