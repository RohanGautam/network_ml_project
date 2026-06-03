"""Cache MART's K=20 predictions for the train+val splits.

Runs MART forward once over deterministic multi-window views of train and val
(using WindowEvalSampler with 8 windows/seq) and saves to disk so the head
selector can iterate on cheap reads instead of re-running MART each epoch.

For each window, saves the 11 real entities only (hoops dropped if present):
    past:        [N=11, T_p, 2]      z-scored past positions
    target:      [N=11, T_f, 2]      z-scored future positions (GT)
    k_preds:     [N=11, K, T_f, 2]   z-scored K hypothesis predictions
    agent_ids:   [N=11]              canonical ids (0=TeamA, 1=TeamB, 2=Ball)

Plus the (mu, sigma) from the checkpoint so the trainer can denormalize.

Usage:
    python cache_mart_preds.py \\
        --checkpoint checkpoints/mart_minade_s1.ckpt \\
        --cache_dir ../../cache/mart_minade_s1 \\
        --split_path ../../splits/fold0.json
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from box import Box
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from models.mart import MART  # noqa: E402
from models.mart_id import MART_ID  # noqa: E402
from loaders.dataloader_nba_pt import (  # noqa: E402
    MARTNBAPTDataset,
    WindowEvalSampler,
    load_split_files,
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
        n_entity_types = (HOOPS_ID + 1) if opts.get('use_hoops', False) else 3
        return MART_ID(opts, num_entity_types=n_entity_types).to(device)
    return MART(opts).to(device)


def cache_split(model, opts, files, mu, sigma, batch_size, num_workers,
                device, use_hoops, windows_per_seq):
    DatasetCls = MARTNBAPTDatasetHoops if use_hoops else MARTNBAPTDataset
    ds = DatasetCls(files, mu, sigma, opts.past_length, opts.future_length)
    sampler = WindowEvalSampler(ds.max_start, windows_per_seq=windows_per_seq)
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    past_c, tgt_c, kp_c, id_c = [], [], [], []
    with torch.no_grad():
        for x_abs, y, agent_ids in loader:
            x_abs, y, agent_ids = (
                x_abs.to(device), y.to(device), agent_ids.to(device)
            )
            x_rel = _x_rel(x_abs)
            if isinstance(model, MART_ID):
                y_pred = model(x_abs, x_rel, agent_ids)
            else:
                y_pred = model(x_abs, x_rel)
            if opts.pred_rel:
                cur = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur
            if use_hoops:
                x_abs = x_abs[:, :HOOPS_N_REAL_AGENTS]
                y = y[:, :HOOPS_N_REAL_AGENTS]
                y_pred = y_pred[:, :HOOPS_N_REAL_AGENTS]
                agent_ids = agent_ids[:, :HOOPS_N_REAL_AGENTS]
            past_c.append(x_abs.cpu())
            tgt_c.append(y.cpu())
            kp_c.append(y_pred.cpu())
            id_c.append(agent_ids.cpu())
    return {
        'past': torch.cat(past_c),         # [M, 11, T_p, 2]
        'target': torch.cat(tgt_c),        # [M, 11, T_f, 2]
        'k_preds': torch.cat(kp_c),        # [M, 11, K, T_f, 2]
        'agent_ids': torch.cat(id_c),      # [M, 11]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split_path', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--windows_per_seq', type=int, default=8)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}  ckpt: {args.checkpoint}')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    mu = ckpt['mu'].float()
    sigma = ckpt['sigma'].float()
    use_hoops = bool(opts.get('use_hoops', False))

    model = _build(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f'[INFO] K={opts.sample_k}, past={opts.past_length}, '
          f'future={opts.future_length}, use_hoops={use_hoops}')

    train_files, val_files = load_split_files(args.split_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name, files in [('train', train_files), ('val', val_files)]:
        print(f'[INFO] caching {name}: {len(files)} sequences, '
              f'windows_per_seq={args.windows_per_seq}')
        c = cache_split(
            model, opts, files, mu, sigma, args.batch_size, args.num_workers,
            device, use_hoops, args.windows_per_seq,
        )
        c['mu'] = mu
        c['sigma'] = sigma
        out = cache_dir / f'{name}.pt'
        torch.save(c, out)
        print(f'  wrote {out}  past:{tuple(c["past"].shape)} '
              f'k_preds:{tuple(c["k_preds"].shape)}')


if __name__ == '__main__':
    main()
