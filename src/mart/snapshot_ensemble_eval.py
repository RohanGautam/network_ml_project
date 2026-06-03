"""Evaluate the SGDR snapshot ensemble: average per-cycle-minimum predictions.

The SGDR run saved one checkpoint at each cosine cycle's minimum
(checkpoints/<name>_snapshots/cycle_NN.ckpt). A snapshot ensemble averages their
predictions — a free same-arch ensemble from a single training run. This only
helps if the cycle minima are DIVERSE solutions of comparable quality; here the
cycle minima improved monotonically (3.56 -> 3.12), so the early ones may be just
weaker (not diverse) and drag the mean up. We test directly:

  1. each snapshot's solo val/mse_ft (mean-of-K, feet, 11 entities),
  2. full 10-snapshot ensemble,
  3. "best-N" ensembles (average only the N best snapshots) — finds whether
     dropping the weak early cycles helps.

Runs MART forward per snapshot on the same WindowEvalSampler val windows used
everywhere else, so numbers are directly comparable to the 3.112 baseline.

Usage:
    python snapshot_ensemble_eval.py \\
        --snapshot_dir checkpoints/mart_aug_iso_hoops_sgdr_snapshots \\
        --split_path ../../splits/fold0.json
"""

import argparse
import glob
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
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID as HOOPS_ID,
)

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


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


def predict_meanK_ft(ckpt_path, val_files, device):
    """Return (pred_ft [M,11,12,2], target_ft [M,11,12,2]) for one snapshot."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    mu, sigma = ckpt['mu'].float(), ckpt['sigma'].float()
    use_hoops = bool(opts.get('use_hoops', False))
    model = _build(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    DatasetCls = MARTNBAPTDatasetHoops if use_hoops else MARTNBAPTDataset
    ds = DatasetCls(val_files, mu, sigma, opts.past_length, opts.future_length)
    sampler = WindowEvalSampler(ds.max_start, windows_per_seq=8)
    loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=4)

    mu_b = mu.view(1, 1, 1, 2).to(device)
    sig_b = sigma.view(1, 1, 1, 2).to(device)
    preds, tgts = [], []
    with torch.no_grad():
        for x_abs, y, agent_ids in loader:
            x_abs, y, agent_ids = x_abs.to(device), y.to(device), agent_ids.to(device)
            x_rel = _x_rel(x_abs)
            yp = model(x_abs, x_rel, agent_ids) if isinstance(model, MART_ID) \
                else model(x_abs, x_rel)
            if opts.pred_rel:
                yp = torch.cumsum(yp, dim=3) + x_abs[:, :, [-1]].unsqueeze(2)
            if use_hoops:
                y = y[:, :HOOPS_N_REAL_AGENTS]
                yp = yp[:, :HOOPS_N_REAL_AGENTS]
            yp_ft = yp.mean(dim=2) * sig_b + mu_b      # mean-of-K -> feet
            y_ft = y * sig_b + mu_b
            preds.append(yp_ft.cpu())
            tgts.append(y_ft.cpu())
    return torch.cat(preds), torch.cat(tgts)


def mse_ft(pred, tgt):
    return ((pred - tgt) ** 2).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot_dir', required=True)
    p.add_argument('--split_path', required=True)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_files = load_split_files(args.split_path)
    snaps = sorted(glob.glob(os.path.join(args.snapshot_dir, 'cycle_*.ckpt')))
    print(f'[INFO] {len(snaps)} snapshots, {len(val_files)} val sequences\n')

    all_preds, target = [], None
    print('=== Per-snapshot solo val/mse_ft ===')
    solo = []
    for s in snaps:
        pred, tgt = predict_meanK_ft(s, val_files, device)
        if target is None:
            target = tgt
        all_preds.append(pred)
        m = mse_ft(pred, target)
        solo.append(m)
        print(f'  {os.path.basename(s):16s} {m:.4f}')

    preds = torch.stack(all_preds)  # [S, M, 11, 12, 2]

    print('\n=== Full ensemble (all snapshots averaged) ===')
    print(f'  all-{len(snaps)}: {mse_ft(preds.mean(0), target):.4f}')

    print('\n=== Best-N ensemble (average only the N best snapshots) ===')
    order = sorted(range(len(snaps)), key=lambda i: solo[i])  # best-first
    best_overall = (None, 1e9)
    for n in range(1, len(snaps) + 1):
        idx = order[:n]
        m = mse_ft(preds[idx].mean(0), target)
        tag = ' '.join(os.path.basename(snaps[i]).replace('cycle_', 'c').replace('.ckpt', '') for i in idx)
        print(f'  best-{n:2d} ({tag}): {m:.4f}')
        if m < best_overall[1]:
            best_overall = (n, m)
    print(f'\n  BEST: best-{best_overall[0]} ensemble -> {best_overall[1]:.4f}')
    print(f'  vs single-cosine baseline 3.112 | best solo snapshot {min(solo):.4f}')


if __name__ == '__main__':
    main()
