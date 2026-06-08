"""Measure test-time augmentation (TTA) gain for aug-MART on the val fold.

aug-MART is only *approximately* O(2)-invariant (taught via augmentation), NOT
exactly equivariant like EqMotion (whose reflection-TTA was a proven no-op). So
averaging predictions over court symmetries should give a genuine
variance-reduction gain here. We use the D2 group — the court's EXACT symmetries
(identity, flip-x, flip-y, 180° rot) — because: the model was trained invariant
to them, the hoop landmark nodes map to valid hoop positions under them, and each
is an involution (apply the same transform to invert the prediction).

Transforms act in NORMALIZED space about the court center c = -mu/sigma:
    z' = M (z - c) + c,   M ∈ {I, diag(-1,1), diag(1,-1), diag(-1,-1)}
Velocities are recomputed from the transformed positions (as in training), so
they stay consistent. Reports val/mse_ft (mean-of-K, feet, 11 entities) for
no-TTA vs each transform alone vs the 4-way D2 average.

Usage:
    python tta_eval.py --checkpoint checkpoints/mart_aug_iso_hoops_5k_best.ckpt \\
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

# D2 court symmetries as diagonal sign-flips (Mx, My) about the court center.
D2 = {'identity': (1.0, 1.0), 'flip_x': (-1.0, 1.0),
      'flip_y': (1.0, -1.0), 'rot180': (-1.0, -1.0)}


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


def _apply_d2(z, sx, sy, center):
    """z [...,2] normalized -> M(z-c)+c with M=diag(sx,sy). Involution."""
    c = center.view(*([1] * (z.dim() - 1)), 2)
    zc = z - c
    out = torch.stack([sx * zc[..., 0], sy * zc[..., 1]], dim=-1)
    return out + c


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
    center = (-mu / sigma).to(device)              # court center in normed space
    model = _build(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f'[INFO] {os.path.basename(args.checkpoint)} use_hoops={use_hoops} '
          f'center={center.tolist()}')

    _, val_files = load_split_files(args.split_path)
    DatasetCls = MARTNBAPTDatasetHoops if use_hoops else MARTNBAPTDataset
    ds = DatasetCls(val_files, mu, sigma, opts.past_length, opts.future_length)
    loader = DataLoader(ds, batch_size=64,
                        sampler=WindowEvalSampler(ds.max_start, windows_per_seq=8),
                        num_workers=4)
    mu_b = mu.view(1, 1, 1, 2).to(device)
    sig_b = sigma.view(1, 1, 1, 2).to(device)

    # Collect per-transform predictions (feet, 11 entities) + target once.
    per_tf = {k: [] for k in D2}
    tgts = []
    with torch.no_grad():
        for x_abs, y, agent_ids in loader:
            x_abs, y, agent_ids = x_abs.to(device), y.to(device), agent_ids.to(device)
            for name, (sx, sy) in D2.items():
                xt = _apply_d2(x_abs, sx, sy, center)
                yp = model(xt, _x_rel(xt), agent_ids) if isinstance(model, MART_ID) \
                    else model(xt, _x_rel(xt))
                if opts.pred_rel:
                    yp = torch.cumsum(yp, dim=3) + xt[:, :, [-1]].unsqueeze(2)
                yp = yp.mean(dim=2)                       # mean-of-K [B,N,T_f,2]
                yp = _apply_d2(yp, sx, sy, center)        # invert (involution)
                if use_hoops:
                    yp = yp[:, :HOOPS_N_REAL_AGENTS]
                per_tf[name].append((yp * sig_b + mu_b).cpu())
            yy = y[:, :HOOPS_N_REAL_AGENTS] if use_hoops else y
            tgts.append((yy * sig_b + mu_b).cpu())

    target = torch.cat(tgts)
    preds = {k: torch.cat(v) for k, v in per_tf.items()}

    def mse(pr):
        return ((pr - target) ** 2).mean().item()

    print('\n=== per-transform val/mse_ft (each should be CLOSE if ~invariant) ===')
    for k in D2:
        print(f'  {k:10s} {mse(preds[k]):.4f}')
    base = mse(preds['identity'])
    d2_avg = mse(sum(preds.values()) / len(preds))
    print(f'\n  no-TTA (identity) : {base:.4f}')
    print(f'  D2 4-way TTA avg  : {d2_avg:.4f}   (Δ {d2_avg - base:+.4f})')
    print(f'  single-cosine baseline reference = 3.112')


if __name__ == '__main__':
    main()
