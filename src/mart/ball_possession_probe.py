"""Is the ball error a *possession* problem? Structural-lever diagnostic.

The error budget says total val ~3.1 is dominated by the ball (~16) while players
are near the ~2.2 floor; closing 3.1->2.6 is almost entirely a ball problem. In
basketball the ball is usually POSSESSED by a player (ball position ~= handler
position) except during passes/shots. If so, the ball's future during possession
is just the handler's future -- a player-difficulty (2.2) problem, not a
free-agent (16) one. The generic model treats the ball as a free node and pays
for it.

For each val window we identify the likely handler = player nearest the ball at
the last context frame (t=C), then compare ball-future predictors (all in feet^2,
ball only):

  model        : the model's own mean-of-K ball prediction (baseline ~16).
  cv           : constant-velocity extrapolation of the ball from its last 2
                 context frames (physics prior, no learning).
  handler_pred : ball := the handler's MODEL-predicted future (realizable: uses
                 only the past). This is the possession-anchor lever.
  handler_gt   : ball := the handler's GROUND-TRUTH future (oracle upper bound on
                 the trackable/possession part; not submittable).

Also reports possession persistence: fraction of horizon steps the ball stays
within R ft of the t=C handler in GT (high => possession dominates => the lever
should pay).

Usage:
    python ball_possession_probe.py \
        --checkpoint checkpoints/mart_aug_iso_hoops_5k_best.ckpt \
        --split_path ../../splits/fold0.json
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from box import Box
from torch.utils.data import DataLoader

from models.mart import MART  # noqa: E402
from models.mart_id import MART_ID  # noqa: E402
from loaders.dataloader_nba_pt import (  # noqa: E402
    MARTNBAPTDataset, WindowEvalSampler, load_split_files,
)
from loaders.dataloader_nba_pt_hoops import (  # noqa: E402
    MARTNBAPTDataset as MARTNBAPTDatasetHoops,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS, HOOP_ID as HOOPS_ID,
)

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

BALL = 10  # canonical order after hoop strip: TeamA(0..4), TeamB(5..9), Ball(10)
R_FT = 6.0  # possession radius


def _x_rel(x_abs):
    r = torch.zeros_like(x_abs)
    r[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    r[:, :, 0] = r[:, :, 1]
    return r


def _build(opts, device):
    if bool(opts.get('use_entity_embed', False)):
        n = (HOOPS_ID + 1) if opts.get('use_hoops', False) else 3
        return MART_ID(opts, num_entity_types=n).to(device)
    return MART(opts).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split_path', required=True)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    mu, sigma = ckpt['mu'].float(), ckpt['sigma'].float()
    use_hoops = bool(opts.get('use_hoops', False))
    assert use_hoops, 'this probe assumes the canonical 11-entity (hoop) order'
    model = _build(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f'[INFO] {os.path.basename(args.checkpoint)} sigma={sigma.tolist()}')

    _, val_files = load_split_files(args.split_path)
    ds = MARTNBAPTDatasetHoops(val_files, mu, sigma, opts.past_length, opts.future_length)
    loader = DataLoader(ds, batch_size=64,
                        sampler=WindowEvalSampler(ds.max_start, windows_per_seq=8),
                        num_workers=4)
    mu_b = mu.view(1, 1, 1, 2).to(device)
    sig_b = sigma.view(1, 1, 1, 2).to(device)

    # squared-error accumulators (ball only) + player/total references
    se = {k: 0.0 for k in ('model_ball', 'cv_ball', 'handler_pred_ball',
                           'handler_gt_ball', 'model_players', 'model_total')}
    n_ball = n_play = n_tot = 0
    persist_sum = 0.0
    persist_n = 0
    handler_is_ball_static = 0  # sanity: how often "nearest" is degenerate
    with torch.no_grad():
        for x_abs, y, agent_ids in loader:
            x_abs, y, agent_ids = x_abs.to(device), y.to(device), agent_ids.to(device)
            yp = model(x_abs, _x_rel(x_abs), agent_ids) if isinstance(model, MART_ID) \
                else model(x_abs, _x_rel(x_abs))
            if opts.pred_rel:
                yp = torch.cumsum(yp, dim=3) + x_abs[:, :, [-1]].unsqueeze(2)
            pred = (yp * sig_b.unsqueeze(2) + mu_b.unsqueeze(2)).mean(dim=2)  # [B,N,T,2] ft
            pred = pred[:, :HOOPS_N_REAL_AGENTS]
            gt = (y[:, :HOOPS_N_REAL_AGENTS] * sig_b + mu_b)                  # [B,N,T,2] ft
            ctx = (x_abs[:, :HOOPS_N_REAL_AGENTS] * sig_b.unsqueeze(2)
                   + mu_b.unsqueeze(2)) if x_abs.dim() == 4 else None
            # x_abs is [B,N,C,2]; denorm to feet
            ctx = x_abs[:, :HOOPS_N_REAL_AGENTS] * sigma.view(1, 1, 1, 2).to(device) \
                + mu.view(1, 1, 1, 2).to(device)
            B, N, T, _ = gt.shape

            # likely handler = player (0..9) nearest the ball at last context frame
            ball_c = ctx[:, BALL, -1, :]                       # [B,2]
            players_c = ctx[:, :10, -1, :]                     # [B,10,2]
            d = torch.norm(players_c - ball_c.unsqueeze(1), dim=-1)  # [B,10]
            handler = d.argmin(dim=1)                          # [B]
            hi = handler.view(B, 1, 1, 1).expand(B, 1, T, 2)

            # predictors for the ball [B,T,2]
            ball_gt = gt[:, BALL]
            model_ball = pred[:, BALL]
            handler_pred_ball = pred[:, :10].gather(1, hi).squeeze(1)
            handler_gt_ball = gt[:, :10].gather(1, hi).squeeze(1)
            v = ctx[:, BALL, -1, :] - ctx[:, BALL, -2, :]      # [B,2] last ball vel
            steps = torch.arange(1, T + 1, device=device).view(1, T, 1)
            cv_ball = ball_c.unsqueeze(1) + v.unsqueeze(1) * steps

            se['model_ball'] += ((model_ball - ball_gt) ** 2).sum().item()
            se['cv_ball'] += ((cv_ball - ball_gt) ** 2).sum().item()
            se['handler_pred_ball'] += ((handler_pred_ball - ball_gt) ** 2).sum().item()
            se['handler_gt_ball'] += ((handler_gt_ball - ball_gt) ** 2).sum().item()
            n_ball += ball_gt.numel()

            se['model_players'] += ((pred[:, :10] - gt[:, :10]) ** 2).sum().item()
            n_play += gt[:, :10].numel()
            se['model_total'] += ((pred - gt) ** 2).sum().item()
            n_tot += gt.numel()

            # possession persistence: ball within R ft of the t=C handler in GT
            handler_gt_track = handler_gt_ball                 # [B,T,2]
            within = (torch.norm(ball_gt - handler_gt_track, dim=-1) <= R_FT).float()
            persist_sum += within.mean(dim=1).sum().item()
            persist_n += B

    print(f'\n=== current split (mean-of-K, feet^2) ===')
    print(f'  total11 : {se["model_total"]/n_tot:.3f}')
    print(f'  players : {se["model_players"]/n_play:.3f}')
    print(f'  ball    : {se["model_ball"]/n_ball:.3f}')
    print(f'\n=== ball predictors (feet^2, ball only) ===')
    for k in ('model_ball', 'cv_ball', 'handler_pred_ball', 'handler_gt_ball'):
        print(f'  {k:18s} {se[k]/n_ball:.3f}')
    print(f'\n  possession persistence : {persist_sum/persist_n:.3f} '
          f'(mean frac of horizon ball within {R_FT:.0f} ft of t=C handler)')
    # projected total if we swap in the realizable handler-anchored ball
    bp = se['handler_pred_ball'] / n_ball
    pl = se['model_players'] / n_play
    print(f'\n  projected total11 if ball := handler_pred : '
          f'{(10 * pl + bp) / 11:.3f}')
    bo = se['handler_gt_ball'] / n_ball
    print(f'  projected total11 if ball := handler_gt (oracle): '
          f'{(10 * pl + bo) / 11:.3f}')


if __name__ == '__main__':
    main()
