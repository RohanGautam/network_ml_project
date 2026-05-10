"""Generate Kaggle submission CSV from a trained GroupNetWithID checkpoint.

For each .pt file in <test_dir>:
    - reorder agents into canonical [TeamA(5), TeamB(5), Ball(1)] (same as training),
      tracking the inverse permutation so we can map predictions back
    - z-score (x, y) with the (mu, sigma) saved inside the checkpoint
    - run model.inference (returns K stochastic samples)
    - reduce K -> 1 (mean by default; optimal for MSE-style losses)
    - denormalize, un-reorder agents back to the test file's original layout
    - flatten to a CSV row in (t, i, axis) order

The CSV columns match the Kaggle sample_submission.csv:
    id, entity_0_time_0_x, entity_0_time_0_y, entity_1_time_0_x, ...,
        entity_10_time_0_y, entity_0_time_1_x, ..., entity_10_time_11_y
(11 entities * 12 future steps * 2 axes = 264 prediction columns).

Example:
    python submit_nba_pt.py \
        --checkpoint saved_models/nba_pt_run1/100.p \
        --test_dir ../network_ml_project/data/test/test \
        --out_csv submissions/submission_run1.csv \
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import pandas as pd
import torch

from model.GroupNet_nba_id import GroupNetWithID


if not torch.cuda.is_available():
    # GroupNet's upstream code (Decoder.forward, MS_HGNN_*) has 8+ hardcoded
    # .cuda() calls scattered across multiple files, which crash on CPU-only
    # machines. Rather than monkey-patching each method, we neutralize the call
    # itself: on CPU, Tensor.cuda() becomes a no-op so the tensor just stays on
    # its current (CPU) device. This override is process-local — training and
    # optuna scripts that don't import this file are unaffected — and is a no-op
    # itself when CUDA is available (this branch never runs).
    print('CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat')
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to a .p checkpoint produced by train_hyper_nba_pt.py')
    p.add_argument('--test_dir', type=str, required=True,
                   help='Directory of test .pt files (e.g. .../data/test/test)')
    p.add_argument('--out_csv', type=str, default='submission.csv')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--reduce', choices=['mean', 'first'], default='mean',
                   help='How to reduce K stochastic samples: mean (best for MSE) '
                        'or just the first sample (more stochastic)')
    p.add_argument('--sample_k', type=int, default=None,
                   help='Override the number of stochastic samples drawn at inference. '
                        'If unset, uses the value from the checkpoint cfg (default 20). '
                        'Higher = smoother mean estimate, more GPU memory.')
    p.add_argument('--sample_k_chunk', type=int, default=50,
                   help='Cap on samples per single decoder pass. If --sample_k > this, '
                        'inference is run in multiple chunks (i.i.d. from the prior) '
                        'and concatenated. Lets you go to K=1000+ without OOM.')
    return p.parse_args()


def inference_chunked(model, data, total_k, chunk_k):
    """Run model.inference() in chunks and concatenate. Statistically equivalent
    to a single pass with K=total_k, since each chunk samples i.i.d. from N(0, I).
    """
    if total_k <= chunk_k:
        model.args.sample_k = total_k
        return model.inference(data)
    chunks = []
    done = 0
    while done < total_k:
        k_this = min(chunk_k, total_k - done)
        model.args.sample_k = k_this
        chunks.append(model.inference(data))   # [k_this, B*N, T_f, 2]
        done += k_this
    return torch.cat(chunks, dim=0)             # [total_k, B*N, T_f, 2]


def reorder_test_seq(seq):
    """Reorder agents to canonical layout and return the inverse permutation.

    Returns:
        reordered: [T, 11, F] in [TeamA(5), TeamB(5), Ball(1)] order
        agent_ids: [11] long, embedding indices for canonical order
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
    agent_ids = torch.tensor([0] * 5 + [1] * 5 + [2], dtype=torch.long)
    return reordered, agent_ids, inv_order


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
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt['model_cfg']
    mu = ckpt['mu'].float()       # [2]
    sigma = ckpt['sigma'].float() # [2]

    # Allow CLI to override how many stochastic samples to draw at inference.
    # K is read by GroupNetWithID.inference() from cfg.sample_k — no model
    # surgery needed, since decoder is fully agnostic to K.
    if args.sample_k is not None:
        print(f'overriding sample_k: {cfg.sample_k} -> {args.sample_k}')
        cfg.sample_k = args.sample_k

    print(f'cfg: past={cfg.past_length}, future={cfg.future_length}, '
          f'embed_dim={cfg.embed_dim}, sample_k={cfg.sample_k}')
    print(f'mu={mu.tolist()}, sigma={sigma.tolist()}')

    # ---- Build model ----
    model = GroupNetWithID(cfg, device, embed_dim=cfg.embed_dim)
    model.load_state_dict(ckpt['model_dict'])
    model.set_device(device)
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

            past_list, ids_list, inv_list, fnames = [], [], [], []
            for f in batch_files:
                seq = torch.load(f).float()                          # [T, 11, F]
                reordered, agent_ids, inv_order = reorder_test_seq(seq)
                xy = (reordered[:, :, :2] - mu) / sigma              # [T, 11, 2]
                xy = xy.permute(1, 0, 2).contiguous()                # [11, T, 2]
                if xy.shape[1] < cfg.past_length:
                    raise ValueError(
                        f'{f}: only {xy.shape[1]} frames, need {cfg.past_length}'
                    )
                past = xy[:, -cfg.past_length:]                      # [11, T_p, 2]
                past_list.append(past)
                ids_list.append(agent_ids)
                inv_list.append(inv_order)
                fnames.append(int(f.stem))

            past = torch.stack(past_list, dim=0)                     # [B, 11, T_p, 2]
            ids = torch.stack(ids_list, dim=0)                       # [B, 11]
            data = {'past_traj': past, 'agent_ids': ids, 'seq': 'nba'}

            pred = inference_chunked(
                model, data,
                total_k=cfg.sample_k,
                chunk_k=args.sample_k_chunk,
            )                                                        # [K, B*11, T_f, 2]
            B = past.shape[0]
            N = past.shape[1]
            T_f = cfg.future_length
            K = pred.shape[0]
            pred = pred.view(K, B, N, T_f, 2).cpu()

            if args.reduce == 'mean':
                pred = pred.mean(dim=0)                              # [B, 11, T_f, 2]
            else:
                pred = pred[0]                                       # [B, 11, T_f, 2]

            # Denormalize: broadcast mu/sigma over last dim (axis x/y).
            pred = pred * sigma + mu                                 # [B, 11, T_f, 2]

            # Un-reorder agents back to the test file's original layout, then
            # permute to (T_f, 11, 2) so that the row's flatten order matches
            # the column header order: t outer, i middle, axis inner.
            for b in range(B):
                pred_raw = pred[b].index_select(0, inv_list[b])      # [11, T_f, 2]
                pred_csv = pred_raw.permute(1, 0, 2).contiguous()    # [T_f, 11, 2]
                flat = pred_csv.reshape(-1).tolist()                 # 264 values
                rows.append([fnames[b]] + flat)

            print(f'  batch {b_idx + 1}/{n_batches}  '
                  f'({len(rows)}/{len(test_files)} rows)')

    # ---- Build DataFrame and write CSV ----
    cols = ['id'] + [
        f'entity_{i}_time_{t}_{axis}'
        for t in range(cfg.future_length)
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
