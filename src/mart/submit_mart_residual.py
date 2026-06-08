"""Generate Kaggle submission using EqMotion + MART-residual for players.

Reads the test cache (EqMotion ensemble predictions on test, in feet, canonical
ordered) and a trained MART residual checkpoint. For each test sequence:
    players (canonical indices 0..9): pos = eqm_pred_ft + MART_residual_ft
    ball (canonical index 10):        pos = eqm_pred_ft (untouched)
Then un-permutes to the test file's original entity layout and writes CSV.

Usage:
    python submit_mart_residual.py \\
        --cache_dir cache/eqm_residual \\
        --mart_residual_ckpt checkpoints/mart_residual_v1.pt \\
        --out_csv submissions/solution_mart_residual.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from box import Box

sys.path.append(os.getcwd())

from models.mart_id import MART_ID  # noqa: E402

PLAYER_IDX = slice(0, 10)
N_ENTITIES = 11
HORIZON = 12


if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def _x_rel(x_abs):
    r = torch.zeros_like(x_abs)
    r[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    r[:, :, 0] = r[:, :, 1]
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--mart_residual_ckpt', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load test cache + residual MART ----
    test = torch.load(Path(args.cache_dir) / 'test.pt', weights_only=False)
    mart_mu = test['mart_mu'].to(device)
    mart_sigma = test['mart_sigma'].to(device)
    print(f'[INFO] test windows: {test["past"].shape[0]}')

    ckpt = torch.load(args.mart_residual_ckpt, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    model = MART_ID(opts, num_entity_types=3).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f'[INFO] residual MART loaded, K={opts.sample_k}')

    # ---- Inference, batched ----
    M = test['past'].shape[0]
    rows = []
    with torch.no_grad():
        for s in range(0, M, args.batch_size):
            past = test['past'][s:s + args.batch_size].to(device)
            eqm_pred_ft = test['eqm_pred_ft'][s:s + args.batch_size].to(device)
            agent_ids = test['agent_ids'][s:s + args.batch_size].to(device)
            inv_order = test['inv_order'][s:s + args.batch_size]
            ids = test['test_ids'][s:s + args.batch_size]

            x_rel = _x_rel(past)
            y_pred = model(past, x_rel, agent_ids)        # [B, N, K, T_f, 2] z
            mean_residual_z = y_pred.mean(dim=2)           # [B, N, T_f, 2]
            mean_residual_ft = mean_residual_z * mart_sigma  # residual: no mu

            combined_ft = eqm_pred_ft.clone()
            combined_ft[:, PLAYER_IDX] += mean_residual_ft[:, PLAYER_IDX]
            # Ball stays at EqMotion base (no residual added).

            # Un-permute canonical -> original test-file order, then flatten
            # in (t, i, axis) order to match Kaggle's column header.
            combined_ft = combined_ft.cpu()
            for b in range(combined_ft.shape[0]):
                pred_canon = combined_ft[b]                 # [11, T_f, 2]
                pred_raw = pred_canon[inv_order[b]]         # [11, T_f, 2] original order
                pred_csv = pred_raw.permute(1, 0, 2)        # [T_f, 11, 2]
                rows.append([int(ids[b].item())] + pred_csv.reshape(-1).tolist())

    cols = ['id'] + [
        f'entity_{i}_time_{t}_{axis}'
        for t in range(HORIZON)
        for i in range(N_ENTITIES)
        for axis in ['x', 'y']
    ]
    df = pd.DataFrame(rows, columns=cols).set_index('id').sort_index()
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f'[INFO] wrote {out}  shape={df.shape}')


if __name__ == '__main__':
    main()
