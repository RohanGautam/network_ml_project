"""Generate Kaggle submission CSV from a trained MART / MART_ID checkpoint.

Mirrors GroupNet/groupnet/submit_nba_pt.py: for each .pt file in <test_dir>,
    - reorder agents into canonical [TeamA(5), TeamB(5), Ball(1)], remembering
      the inverse permutation so predictions map back to the test file's order
    - z-score (x, y) with the (mu, sigma) saved in the checkpoint
    - compute x_rel (velocity) from x_abs the same way training does
    - run model(x_abs, x_rel[, agent_ids]) -> [B, N, K, T_f, 2]
    - reduce K -> 1 (mean by default; optimal for MSE-style scoring)
    - denormalize and un-reorder agents back to the original layout
    - flatten to a CSV row in (t, i, axis) order

Difference vs GroupNet's submit: MART's K decoder heads are deterministic and
baked into the model architecture, so there's no sample_k override and no
chunked inference. K is whatever the model was trained with.

CSV columns match Kaggle sample_submission.csv:
    id, entity_0_time_0_x, entity_0_time_0_y, ..., entity_10_time_<T_f-1>_y
(11 entities * future_length steps * 2 axes prediction columns).

Example:
    python submit_nba_pt.py \\
        --checkpoint checkpoints/mart_pt_run1.ckpt \\
        --test_dir ../network_ml_project/data/test/test \\
        --out_csv submissions/mart_pt_run1.csv \\
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import pandas as pd
import torch
from box import Box

from models.mart import MART
from models.mart_id import MART_ID
from loaders.dataloader_nba_pt_hoops import (
    RAW_HOOPS as HOOPS_RAW,
    N_LANDMARKS as HOOPS_N_LANDMARKS,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID as HOOPS_ID,
)


if not torch.cuda.is_available():
    # MART has 3 hardcoded .cuda() calls in prt.py / hrt.py for relation
    # matrices it builds on the fly. Neutralize them on CPU so the same
    # checkpoint runs anywhere. No-op when CUDA is available.
    print('CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat')
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


# Canonical embedding indices after the TeamA / TeamB / Ball reorder.
CANONICAL_AGENT_IDS = torch.tensor([0] * 5 + [1] * 5 + [2], dtype=torch.long)
# Same layout extended with 2 basket-hoop landmarks (id=HOOPS_ID). Used when the
# checkpoint was trained with --use_hoops.
CANONICAL_AGENT_IDS_HOOPS = torch.tensor(
    [0] * 5 + [1] * 5 + [2] + [HOOPS_ID] * HOOPS_N_LANDMARKS, dtype=torch.long,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to a .ckpt produced by main_nba_pt.py')
    p.add_argument('--test_dir', type=str, required=True,
                   help='Directory of test .pt files (e.g. .../data/test/test)')
    p.add_argument('--out_csv', type=str, default='submission.csv')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--reduce', choices=['mean', 'first'], default='mean',
                   help='How to reduce MART\'s K decoder heads to one prediction: '
                        'mean (best for MSE) or just the first head.')
    return p.parse_args()


def reorder_test_seq(seq):
    """Reorder agents to canonical layout and return the inverse permutation.

    Returns:
        reordered: [T, 11, F] in [TeamA(5), TeamB(5), Ball(1)] order
        inv_order: [11] long, such that pred_raw[i] = pred_canonical[inv_order[i]]
    """
    ids = seq[0, :, -1].long()
    team_a = (ids == -1).nonzero(as_tuple=True)[0]
    team_b = (ids == 1).nonzero(as_tuple=True)[0]
    ball = (ids == 0).nonzero(as_tuple=True)[0]
    if len(team_a) != 5 or len(team_b) != 5 or len(ball) != 1:
        raise ValueError(
            f"Unexpected agent layout: 5/5/1 expected, got "
            f"{len(team_a)}/{len(team_b)}/{len(ball)}; ids={ids.tolist()}"
        )
    new_order = torch.cat([team_a, team_b, ball])
    inv_order = torch.argsort(new_order)
    reordered = seq.index_select(1, new_order)
    return reordered, inv_order


def _x_rel_from_x_abs(x_abs):
    """Same convention as main_nba_pt.py / MART/main_nba.py."""
    x_rel = torch.zeros_like(x_abs)
    x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    x_rel[:, :, 0] = x_rel[:, :, 1]
    return x_rel


def build_model(opts, device):
    """Re-instantiate the architecture exactly as it was trained."""
    use_id = bool(opts.get('use_entity_embed', False))
    if use_id:
        # Match the checkpoint's entity-embedding size. With --use_hoops the
        # basket class (id=HOOPS_ID) adds a row; building with the default
        # 3 classes would fail to load weights or index out-of-bounds.
        n_entity_types = (HOOPS_ID + 1) if opts.get('use_hoops', False) else 3
        print(
            f'[INFO] rebuilding MART_ID with embed_dim={opts.embed_dim}, '
            f'num_entity_types={n_entity_types}'
        )
        model = MART_ID(opts, num_entity_types=n_entity_types).to(device)
    else:
        print('[INFO] rebuilding stock MART (no entity embedding)')
        model = MART(opts).to(device)
    return model, use_id


def main():
    args = parse_args()
    device = (
        torch.device('cuda', args.gpu)
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    print('device:', device)

    # ---- Load checkpoint ----
    print(f'loading checkpoint: {args.checkpoint}')
    # weights_only=False because we save a Box config object alongside the
    # state_dict (PyTorch 2.6 defaults to weights_only=True and would reject it).
    # Safe: this is our own checkpoint, not third-party.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    opts = Box(ckpt['opts'])
    mu = ckpt['mu'].float()       # [2]
    sigma = ckpt['sigma'].float() # [2]

    use_hoops = bool(opts.get('use_hoops', False))
    print(
        f'cfg: past={opts.past_length}, future={opts.future_length}, '
        f'model_dim={opts.model_dim}, sample_k={opts.sample_k}, '
        f'use_entity_embed={opts.get("use_entity_embed", False)}, '
        f'use_hoops={use_hoops}'
    )
    print(f'mu={mu.tolist()}, sigma={sigma.tolist()}')

    # Pre-normalize the basket landmark positions once when needed.
    # Same coords/normalization as MART/loaders/dataloader_nba_pt_hoops.py.
    if use_hoops:
        norm_hoops = ((HOOPS_RAW - mu.view(1, 2)) / sigma.view(1, 2)).float()  # [2, 2]
    else:
        norm_hoops = None

    # ---- Build model and load weights ----
    model, use_id = build_model(opts, device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    # ---- Enumerate test files (sorted by integer id from filename) ----
    test_files = sorted(
        Path(args.test_dir).glob('*.pt'),
        key=lambda p: int(p.stem),
    )
    if len(test_files) == 0:
        raise RuntimeError(f'no .pt files in {args.test_dir}')
    print(f'found {len(test_files)} test files')

    # ---- Inference loop ----
    rows = []
    bsz = args.batch_size
    n_batches = (len(test_files) + bsz - 1) // bsz

    with torch.no_grad():
        for b_idx in range(n_batches):
            batch_files = test_files[b_idx * bsz: (b_idx + 1) * bsz]

            past_list, inv_list, fnames = [], [], []
            for f in batch_files:
                seq = torch.load(f).float()                          # [T, 11, F]
                reordered, inv_order = reorder_test_seq(seq)
                xy = (reordered[:, :, :2] - mu) / sigma              # [T, 11, 2]
                xy = xy.permute(1, 0, 2).contiguous()                # [11, T, 2]
                if xy.shape[1] < opts.past_length:
                    raise ValueError(
                        f'{f}: only {xy.shape[1]} frames, need {opts.past_length}'
                    )
                past = xy[:, -opts.past_length:]                     # [11, T_p, 2]

                # If the checkpoint was trained with hoops, append the same 2
                # static landmark nodes the dataloader injects during training:
                # constant normalized position across all T_p past steps.
                if use_hoops:
                    T_p = past.shape[1]
                    hoop_past = (
                        norm_hoops.unsqueeze(1).expand(-1, T_p, -1).contiguous()
                    )                                                # [2, T_p, 2]
                    past = torch.cat([past, hoop_past], dim=0)       # [13, T_p, 2]

                past_list.append(past)
                inv_list.append(inv_order)
                fnames.append(int(f.stem))

            x_abs = torch.stack(past_list, dim=0).to(device)         # [B, N, T_p, 2]
            x_rel = _x_rel_from_x_abs(x_abs)
            B, N = x_abs.shape[:2]

            if use_id:
                ids = CANONICAL_AGENT_IDS_HOOPS if use_hoops else CANONICAL_AGENT_IDS
                agent_ids = ids.unsqueeze(0).expand(B, N).to(device)
                pred = model(x_abs, x_rel, agent_ids)                # [B, N, K, T_f, 2]
            else:
                pred = model(x_abs, x_rel)

            # Optional pred_rel handling (kept for parity with training).
            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                pred = torch.cumsum(pred, dim=3) + cur_pos

            pred = pred.cpu()                                         # [B, N, K, T_f, 2]

            if args.reduce == 'mean':
                pred = pred.mean(dim=2)                               # [B, N, T_f, 2]
            else:
                pred = pred[:, :, 0]                                  # [B, N, T_f, 2]

            # Drop hoop predictions before agent-reorder: inv_order is 11-long
            # (from the test file), and the CSV header is over 11 entities. The
            # hoops were only inputs; the model isn't expected to predict them.
            if use_hoops:
                pred = pred[:, :HOOPS_N_REAL_AGENTS]                  # [B, 11, T_f, 2]

            # Denormalize: broadcast mu/sigma over the last (axis) dim.
            pred = pred * sigma + mu                                  # [B, 11, T_f, 2]

            # Un-reorder agents back to the test file's original layout, then
            # permute to (T_f, N, 2) so the row's flatten order matches the
            # column header order: t outer, i middle, axis inner.
            for b in range(B):
                pred_raw = pred[b].index_select(0, inv_list[b])       # [N, T_f, 2]
                pred_csv = pred_raw.permute(1, 0, 2).contiguous()     # [T_f, N, 2]
                flat = pred_csv.reshape(-1).tolist()
                rows.append([fnames[b]] + flat)

            print(
                f'  batch {b_idx + 1}/{n_batches}  '
                f'({len(rows)}/{len(test_files)} rows)'
            )

    # ---- Build DataFrame and write CSV ----
    cols = ['id'] + [
        f'entity_{i}_time_{t}_{axis}'
        for t in range(opts.future_length)
        for i in range(11)
        for axis in ['x', 'y']
    ]
    df = pd.DataFrame(rows, columns=cols).set_index('id').sort_index()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f'wrote: {out_path}  ({len(df)} rows, {len(cols) - 1} prediction columns)')


if __name__ == '__main__':
    main()
