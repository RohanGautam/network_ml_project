"""Compare inference strategies for a Laplace NLL checkpoint on the val split.

Three strategies evaluated, all reporting val/mse_ft (feet²):

  mean_loc   — uniform mean of all K loc predictions (same as standard eval)
  min_sigma  — pick the single most confident mode (lowest mean scale)
  weighted   — weighted mean of K locs, weights = softmax(-mean_scale)
               (modes with lower uncertainty get higher weight)

Usage:
    python laplace_inference.py \
        --checkpoint checkpoints/mart_laplace_nll_2k_best.ckpt \
        --split_path ../../splits/fold0.json
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from box import Box
from models.mart import MART
from models.mart_id import MART_ID
from loaders.dataloader_nba_pt import load_split_files, compute_xy_stats
from loaders.dataloader_nba_pt_hoops import (
    MARTNBAPTDataset as HoopsDataset,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID,
)
from loaders.dataloader_nba_pt import MARTNBAPTDataset
from torch.utils.data import DataLoader
from main_nba_pt import WindowEvalSampler

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **kw: self


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split_path', default='../../splits/fold0.json')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def _x_rel_from_x_abs(x_abs):
    x_rel = torch.zeros_like(x_abs)
    x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    x_rel[:, :, 0] = x_rel[:, :, 1]
    return x_rel


@torch.no_grad()
def evaluate(model, loader, opts, device, mu, sigma, strategy):
    """Compute val/mse_ft for a given inference strategy.

    strategy options:
        mean_loc  — uniform mean of K loc predictions
        min_sigma — loc of the mode with lowest mean scale (most confident)
        weighted  — weighted mean: weight_k = softmax(-mean_scale_k)
    """
    MIN_SCALE = 1e-3
    model.eval()
    mu_b = mu.to(device).view(1, 1, 1, 2).float()
    sigma_b = sigma.to(device).view(1, 1, 1, 2).float()
    sse, cnt = 0.0, 0

    use_hoops = bool(opts.get('use_hoops', False))
    N_real = HOOPS_N_REAL_AGENTS if use_hoops else 11

    for x_abs, y, agent_ids in loader:
        x_abs = x_abs.to(device)
        y = y.to(device)
        agent_ids = agent_ids.to(device)
        B, N = x_abs.shape[:2]

        x_rel = _x_rel_from_x_abs(x_abs)

        fwd = model(x_abs, x_rel, agent_ids,
                    mu=mu.to(device), sigma=sigma.to(device))
        # CFI returns tuple; take branch2
        raw = fwd[1] if isinstance(fwd, tuple) else fwd   # [B, N, K, T_f, 4]

        loc   = raw[..., :2]                               # [B, N, K, T_f, 2]
        scale = F.softplus(raw[..., 2:]) + MIN_SCALE       # [B, N, K, T_f, 2]

        # ---- apply strategy ----
        if strategy == 'mean_loc':
            pred = loc.mean(dim=2)                         # [B, N, T_f, 2]

        elif strategy == 'min_sigma':
            # Per-mode mean scale (scalar confidence score)
            mean_scale = scale.mean(dim=(-1, -2))          # [B, N, K]
            best_k = mean_scale.argmin(dim=2)              # [B, N]
            idx = best_k.view(B, N, 1, 1, 1).expand(B, N, 1, loc.shape[3], 2)
            pred = loc.gather(2, idx).squeeze(2)           # [B, N, T_f, 2]

        elif strategy == 'weighted':
            # Softmax over negative mean scale → confident modes get higher weight
            mean_scale = scale.mean(dim=(-1, -2))          # [B, N, K]
            weights = F.softmax(-mean_scale, dim=2)        # [B, N, K]
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, N, K, 1, 1]
            pred = (loc * weights).sum(dim=2)              # [B, N, T_f, 2]

        # Strip hoops, denorm, compute MSE
        pred_real = pred[:, :N_real]                       # [B, N_real, T_f, 2]
        y_real    = y[:, :N_real]                          # [B, N_real, T_f, 2]

        pred_ft = pred_real * sigma_b + mu_b
        y_ft    = y_real   * sigma_b + mu_b

        sse += ((pred_ft - y_ft) ** 2).sum().item()
        cnt += pred_ft.numel()

    return sse / cnt


def main():
    args = parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    mu    = ckpt['mu'].float()
    sigma = ckpt['sigma'].float()
    print(f'loaded: {args.checkpoint}')
    print(f'  sample_k={opts.sample_k}  laplace_nll={opts.get("laplace_nll", False)}')
    print(f'  mu={mu.tolist()}  sigma={sigma.tolist()}')

    if not opts.get('laplace_nll', False):
        print('[WARN] checkpoint was not trained with --laplace_nll; '
              'scale channels will be random. Results may be meaningless.')

    # ---- Build model ----
    use_id = bool(opts.get('use_entity_embed', False))
    use_hoops = bool(opts.get('use_hoops', False))
    if use_id:
        n_types = (HOOP_ID + 1) if use_hoops else 3
        model = MART_ID(opts, num_entity_types=n_types).to(device)
    else:
        model = MART(opts).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    print(f'model params: {sum(p.numel() for p in model.parameters()):,}')

    # ---- Data ----
    train_files, val_files = load_split_files(args.split_path)
    mu_data, sigma_data = compute_xy_stats(train_files, iso=bool(opts.get('iso_norm', False)))

    DatasetCls = HoopsDataset if use_hoops else MARTNBAPTDataset
    val_set = DatasetCls(val_files, mu_data, sigma_data,
                         opts.past_length, opts.future_length)
    val_sampler = WindowEvalSampler(val_set.max_start, windows_per_seq=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())
    print(f'val windows: {len(val_sampler)}')

    # ---- Evaluate all strategies ----
    print('\n=== Laplace NLL Inference Comparison ===')
    for strategy in ('mean_loc', 'min_sigma', 'weighted'):
        mse = evaluate(model, val_loader, opts, device, mu_data, sigma_data, strategy)
        print(f'  {strategy:12s}  val/mse_ft = {mse:.4f} ft²')

    print()


if __name__ == '__main__':
    main()
