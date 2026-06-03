"""Find the optimal best-3-snapshot-ensemble + EqMotion blend weight on val.

The 0.70/0.30 production blend tuned w against SINGLE aug-MART. The best-3 SGDR
snapshot ensemble is a different (lower-variance) predictor, so its optimal blend
weight with EqMotion may differ. This computes best-3's mean-of-K val predictions
(feet), pulls the cached EqMotion ensemble val predictions, verifies window
alignment, and sweeps w in w*best3 + (1-w)*EqMotion to report the optimal weight
+ val/mse_ft. The chosen weight is then applied to the TEST CSV blend by the job.

Usage:
    python tune_best3_eqm_blend.py \\
        --snapshot_dir checkpoints/mart_aug_iso_hoops_sgdr_snapshots \\
        --cycles 07 08 09 \\
        --eqm_cache ../../cache/eqm_residual \\
        --split_path ../../splits/fold0.json
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch

from snapshot_ensemble_eval import predict_meanK_ft  # reuse the validated path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot_dir', required=True)
    p.add_argument('--cycles', nargs='+', default=['07', '08', '09'])
    p.add_argument('--eqm_cache', default='../../cache/eqm_residual')
    p.add_argument('--split_path', required=True)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from loaders.dataloader_nba_pt import load_split_files
    _, val_files = load_split_files(args.split_path)

    # best-3 ensemble val preds (mean of the cycle members' mean-of-K), in feet.
    members, target = [], None
    for c in args.cycles:
        ckpt = os.path.join(args.snapshot_dir, f'cycle_{c}.ckpt')
        pred, tgt = predict_meanK_ft(ckpt, val_files, device)
        if target is None:
            target = tgt
        members.append(pred)
    best3 = torch.stack(members).mean(0)  # [M,11,12,2] feet

    # EqMotion ensemble val preds (feet) from cache; target there is normalized.
    eqm = torch.load(os.path.join(args.eqm_cache, 'val.pt'), weights_only=False)
    eqm_mu = eqm['mart_mu'].view(1, 1, 1, 2)
    eqm_sig = eqm['mart_sigma'].view(1, 1, 1, 2)
    tgt_eqm = eqm['target'] * eqm_sig + eqm_mu
    eqm_ft = eqm['eqm_pred_ft']

    md = (target - tgt_eqm).abs().max().item()
    print(f'[ALIGN] best3 target vs eqm target (feet) max diff: {md:.4f}')
    if md > 1e-2:
        sys.exit('[ALIGN] windows not aligned — abort.')

    def mse(p):
        return ((p - target) ** 2).mean().item()

    print(f'\n  best-3 alone : {mse(best3):.4f}')
    print(f'  EqMotion alone: {mse(eqm_ft):.4f}')
    print('\n=== sweep w*best3 + (1-w)*EqMotion ===')
    best = (None, 1e9)
    for w in np.linspace(0.0, 1.0, 21):
        m = mse(w * best3 + (1 - w) * eqm_ft)
        if m < best[1]:
            best = (round(float(w), 2), m)
    for w in np.linspace(max(0, best[0] - 0.1), min(1, best[0] + 0.1), 21):
        m = mse(w * best3 + (1 - w) * eqm_ft)
        if m < best[1]:
            best = (round(float(w), 3), m)
    print(f'  OPTIMAL w_best3 = {best[0]} -> val/mse_ft {best[1]:.4f}')
    print(f'  (production single-augMART blend was w=0.70 -> 3.073)')
    # Emit the weight on a parseable line for the job to capture.
    print(f'CHOSEN_WEIGHT={best[0]}')


if __name__ == '__main__':
    main()
