"""Sweep K (number of stochastic samples) on the validation split and report
ADE / FDE (in court units) for each. Use this to pick the best K for your
Kaggle submission without wasting daily-submission quota.

Example:
    python eval_k_sweep.py \
        --checkpoint saved_models/nba_pt_run1/100.p \
        --split_path ../network_ml_project/splits/fold0.json \
        --ks 20 50 100 200 500 1000 \
        --batch_size 8 \
        --sample_k_chunk 50 \
        --gpu 0
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader

from data.dataloader_nba_pt import (
    GroupNetNBAPTDataset,
    WindowSampler,
    groupnet_collate,
    load_split_files,
)
from model.GroupNet_nba_id import GroupNetWithID
from submit_nba_pt import inference_chunked


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--split_path', type=str, required=True)
    p.add_argument('--ks', type=int, nargs='+',
                   default=[20, 50, 100, 200, 500])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--sample_k_chunk', type=int, default=50,
                   help='Max samples per decoder pass; large K is split into chunks.')
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = (
        torch.device('cuda', args.gpu) if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt['model_cfg']
    mu = ckpt['mu'].float()
    sigma = ckpt['sigma'].float()

    # Build model
    model = GroupNetWithID(cfg, device, embed_dim=cfg.embed_dim)
    model.load_state_dict(ckpt['model_dict'])
    model.set_device(device)
    model.eval()

    # Load val split
    _, val_files = load_split_files(args.split_path)
    val_set = GroupNetNBAPTDataset(
        val_files, mu, sigma, cfg.past_length, cfg.future_length,
    )
    val_sampler = WindowSampler(
        args.batch_size, val_set.max_start, seed=0, shuffle=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=groupnet_collate,
        pin_memory=torch.cuda.is_available(),
    )
    print(f'val sequences: {len(val_set)} | batches: {len(val_loader)}')

    sigma_b = sigma.view(1, 1, 1, 2)
    mu_b = mu.view(1, 1, 1, 2)

    print(f'\n{"K":>6} | {"ADE":>8} | {"FDE":>8}  (court units)')
    print('-' * 32)
    results = {}
    for K in args.ks:
        ade_sum = 0.0
        fde_sum = 0.0
        n_total = 0
        with torch.no_grad():
            for data in val_loader:
                pred = inference_chunked(
                    model, data, total_k=K, chunk_k=args.sample_k_chunk,
                )                                              # [K, B*N, T_f, 2]
                B = data['past_traj'].shape[0]
                N = data['past_traj'].shape[1]
                T_f = cfg.future_length
                pred = pred.view(K, B, N, T_f, 2).cpu()
                pred_mean = pred.mean(dim=0)                    # [B, N, T_f, 2]

                pred_denorm = pred_mean * sigma_b + mu_b        # un-zscore
                target = data['future_traj'] * sigma_b + mu_b   # already on cpu

                err = torch.norm(pred_denorm - target, dim=-1)  # [B, N, T_f]
                ade = err.mean(dim=-1)                          # [B, N]
                fde = err[..., -1]                              # [B, N]
                ade_sum += ade.sum().item()
                fde_sum += fde.sum().item()
                n_total += ade.numel()

        ade_avg = ade_sum / n_total
        fde_avg = fde_sum / n_total
        results[K] = (ade_avg, fde_avg)
        print(f'{K:>6d} | {ade_avg:>8.4f} | {fde_avg:>8.4f}')

    best_k = min(results, key=lambda k: results[k][0])
    print(f'\nbest K by val ADE: {best_k}  '
          f'(ADE={results[best_k][0]:.4f}, FDE={results[best_k][1]:.4f})')


if __name__ == '__main__':
    main()
