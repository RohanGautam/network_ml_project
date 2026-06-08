"""Quantify the mode-aggregation headroom for an aug-MART checkpoint on val.

The big-model run exposed that our `min_ade` (best-of-K) training objective is at
odds with the submission metric (mean-of-K MSE in feet). This probe measures, on
the SAME K=20 samples from a trained checkpoint, how much of the gap to the
leaderboard is recoverable by *aggregating the modes differently* — with zero
retraining:

  mean-of-K  : what we currently submit (baseline ~3.11).
  medoid     : the actual sample closest to all the others (robust central mode).
  dense_ctr  : centroid of the densest cluster -- pick the sample with the most
               neighbours within a per-entity radius, average that neighbourhood.
               This is the "k-means densest cluster" idea: it only beats the plain
               mean if the modes are genuinely SPREAD (else dense_ctr == mean).
  oracle-ent : per-entity best sample vs GT  -> the ceiling if we could pick the
               right mode for each entity at test time (NOT submittable, a bound).

It also reports sample DIVERSITY (mean per-position std across the K samples, in
feet): near 0 => mode-collapsed (clustering is pointless, the small-model regime);
large => modes are spread (clustering / densest-cluster can help).

All in feet^2 over the 11 real entities, mean-of-K denormalized like training.

Usage:
    python agg_probe.py --checkpoint checkpoints/mart_aug_iso_hoops_5k_best.ckpt \
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
    model = _build(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f'[INFO] {os.path.basename(args.checkpoint)} use_hoops={use_hoops} '
          f'sigma={sigma.tolist()}')

    _, val_files = load_split_files(args.split_path)
    DatasetCls = MARTNBAPTDatasetHoops if use_hoops else MARTNBAPTDataset
    ds = DatasetCls(val_files, mu, sigma, opts.past_length, opts.future_length)
    loader = DataLoader(ds, batch_size=64,
                        sampler=WindowEvalSampler(ds.max_start, windows_per_seq=8),
                        num_workers=4)
    mu_b = mu.view(1, 1, 1, 2).to(device)
    sig_b = sigma.view(1, 1, 1, 2).to(device)

    se = {'mean': 0.0, 'medoid': 0.0, 'dense_ctr': 0.0, 'oracle_ent': 0.0}
    div_sum = 0.0       # sum of per-position std across K (feet)
    div_n = 0
    n_elem = 0
    with torch.no_grad():
        for x_abs, y, agent_ids in loader:
            x_abs, y, agent_ids = x_abs.to(device), y.to(device), agent_ids.to(device)
            yp = model(x_abs, _x_rel(x_abs), agent_ids) if isinstance(model, MART_ID) \
                else model(x_abs, _x_rel(x_abs))
            if opts.pred_rel:
                yp = torch.cumsum(yp, dim=3) + x_abs[:, :, [-1]].unsqueeze(2)
            # yp: [B, N, K, T_f, 2]   ;  to feet
            yp = yp * sig_b.unsqueeze(2) + mu_b.unsqueeze(2)
            if use_hoops:
                yp = yp[:, :HOOPS_N_REAL_AGENTS]
                y = y[:, :HOOPS_N_REAL_AGENTS]
            gt = (y * sig_b + mu_b)                          # [B, N, T_f, 2]
            B, N, K, T, _ = yp.shape

            # mean-of-K (what we submit)
            pred_mean = yp.mean(dim=2)                       # [B,N,T,2]

            # diversity: per-position std across the K samples (feet).
            div_sum += yp.std(dim=2).sum().item()
            div_n += B * N * T * 2

            # medoid: sample minimizing sum of L2 dist to the other K-1 samples,
            # per (B,N) over the flattened (T,2) trajectory.
            flat = yp.reshape(B, N, K, T * 2)
            d = torch.cdist(flat, flat)                      # [B,N,K,K]
            medoid_idx = d.sum(dim=-1).argmin(dim=-1)        # [B,N]
            mi = medoid_idx.view(B, N, 1, 1, 1).expand(B, N, 1, T, 2)
            pred_medoid = yp.gather(2, mi).squeeze(2)        # [B,N,T,2]

            # densest-cluster centroid: radius = per-entity median pairwise dist;
            # pick the sample with the most neighbours within r, average that
            # neighbourhood. Collapses to the plain mean when modes aren't spread.
            r = d.median(dim=-1).values.median(dim=-1, keepdim=True).values  # [B,N,1]
            within = (d <= r.unsqueeze(-1)).float()          # [B,N,K,K]
            core = within.sum(dim=-1).argmax(dim=-1)         # [B,N] densest sample
            mask = within.gather(2, core.view(B, N, 1, 1).expand(B, N, 1, K))
            mask = mask.squeeze(2)                           # [B,N,K] neighbourhood
            w = (mask / mask.sum(dim=-1, keepdim=True)).view(B, N, K, 1, 1)
            pred_dense = (yp * w).sum(dim=2)                 # [B,N,T,2]

            # per-entity oracle: sample closest to GT (ceiling, not submittable)
            err_k = ((yp - gt.unsqueeze(2)) ** 2).sum(dim=(3, 4))   # [B,N,K]
            oracle_idx = err_k.argmin(dim=-1)                # [B,N]
            oi = oracle_idx.view(B, N, 1, 1, 1).expand(B, N, 1, T, 2)
            pred_oracle = yp.gather(2, oi).squeeze(2)        # [B,N,T,2]

            for key, pr in (('mean', pred_mean), ('medoid', pred_medoid),
                            ('dense_ctr', pred_dense), ('oracle_ent', pred_oracle)):
                se[key] += ((pr - gt) ** 2).sum().item()
            n_elem += gt.numel()

    print(f'\n  sample diversity : {div_sum / div_n:.3f} ft  '
          f'(per-position std across K; ~0 => mode-collapsed)')
    print('\n=== val/mse_ft by mode aggregation (mean-of-K = baseline) ===')
    for key in ('mean', 'medoid', 'dense_ctr', 'oracle_ent'):
        mse = se[key] / n_elem
        print(f'  {key:11s} {mse:.4f} ft^2   (RMS displ {(2*mse) ** 0.5:.3f} ft)')
    base = se['mean'] / n_elem
    print(f'\n  medoid    Δ vs mean : {se["medoid"]/n_elem - base:+.4f}')
    print(f'  dense_ctr Δ vs mean : {se["dense_ctr"]/n_elem - base:+.4f}')
    print(f'  oracle headroom     : {se["oracle_ent"]/n_elem - base:+.4f}  '
          f'(perfect per-entity mode selection; not submittable)')


if __name__ == '__main__':
    main()
