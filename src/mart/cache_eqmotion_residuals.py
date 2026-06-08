"""Cache EqMotion 5-seed ensemble predictions for residual-MART training.

For each window in train/val/test, runs all 5 EqMotion iso+hoops seeds and
averages their predictions in feet. Stores the result alongside MART's
canonical-order past/target tensors so the residual trainer can iterate
quickly without re-running EqMotion.

Windowing (matches what MART's deterministic eval uses):
    - train, val: 8 evenly-spaced windows/sequence (via WindowEvalSampler logic).
    - test:       single window per sequence (test .pt files are length-8
                  context-only, no horizon, so start=0 is the only window).

Cache layout per split (.pt dict):
    past:        [M, 11, T_p, 2]   z-scored in MART's (aniso) frame, canonical order
    target:      [M, 11, T_f, 2]   same for ground truth (absent for test)
    eqm_pred_ft: [M, 11, T_f, 2]   EqMotion ensemble mean in feet, canonical order
    agent_ids:   [M, 11]           canonical [0]*5 + [1]*5 + [2]
    inv_order:   [M, 11]           inverse permutation back to test-file order
    mu, sigma:   [2]               MART's per-axis normalization stats

Usage:
    python cache_eqmotion_residuals.py \\
        --eqmotion_ckpts checkpoints/eqmotion/iso_hoops_s*/best.ckpt \\
        --mart_ckpt src/mart/checkpoints/mart_minade_s1.ckpt \\
        --split_path splits/fold0.json \\
        --test_dir data/test/test \\
        --cache_dir cache/eqm_residual
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
from box import Box


def _resolve_data_dir(split_path):
    """Read the split manifest and return the absolute data dir."""
    manifest_path = Path(split_path).resolve()
    project_root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())
    data_dir = project_root / manifest['data_dir']
    return data_dir, manifest['train'], manifest['val']


def _starts_for(max_start, k):
    """Evenly-spaced window starts; mirrors NBAEvalSampler / WindowEvalSampler."""
    if max_start <= 0:
        return [0]
    n = min(k, max_start + 1)
    return sorted({int(round(s)) for s in torch.linspace(0, max_start, n).tolist()})


def _build_eqm_input(past_seq, eqm_mu, eqm_sigma, hoops_raw):
    """past_seq [T_p, 11, 4] (raw) -> EqMotion input [T_p, 13, 6] (iso z-scored)."""
    T_p = past_seq.shape[0]
    pos = past_seq[:, :, :2]
    pos_iso = (pos - eqm_mu) / eqm_sigma
    vel = torch.zeros_like(pos_iso)
    vel[1:] = pos_iso[1:] - pos_iso[:-1]
    inp = torch.cat([pos_iso, vel, past_seq[:, :, 2:]], dim=-1)  # [T_p, 11, 6]
    # Append 2 hoops (team_id=3, isplayer=0, zero velocity)
    norm_hoops = (hoops_raw - eqm_mu) / eqm_sigma
    hoop_nodes = torch.zeros((T_p, 2, 6), dtype=inp.dtype)
    hoop_nodes[:, :, :2] = norm_hoops
    hoop_nodes[:, :, 4] = 0.0
    hoop_nodes[:, :, 5] = 3.0
    return torch.cat([inp, hoop_nodes], dim=1)  # [T_p, 13, 6]


def _canonical_remap(raw_seq):
    """Return (canonical_order [11], canonical_agent_ids [11], inv_order [11]).

    raw_seq: [T, 11, 4]. The last feature column holds the entity team id
    (-1 / 0 / 1 for TeamA / Ball / TeamB in the raw layout)."""
    ids = raw_seq[0, :, -1].long()
    team_a = (ids == -1).nonzero(as_tuple=True)[0]
    team_b = (ids == 1).nonzero(as_tuple=True)[0]
    ball = (ids == 0).nonzero(as_tuple=True)[0]
    if len(team_a) != 5 or len(team_b) != 5 or len(ball) != 1:
        raise ValueError(f'unexpected agent layout: {ids.tolist()}')
    canonical_order = torch.cat([team_a, team_b, ball])  # [11], orig idx in canonical order
    inv_order = torch.argsort(canonical_order)            # [11], canonical idx for each orig
    canonical_ids = torch.tensor([0] * 5 + [1] * 5 + [2], dtype=torch.long)
    return canonical_order, canonical_ids, inv_order


def run_eqm_ensemble(eqm_models, eqm_input_batch, eqm_mu, eqm_sigma, device):
    """Average K seed predictions in feet. eqm_input_batch [B, T_p, 13, 6]."""
    B = eqm_input_batch.shape[0]
    eqm_input_batch = eqm_input_batch.to(device)
    preds = []
    for m in eqm_models:
        with torch.no_grad():
            p = m(eqm_input_batch)  # [T_f, B*13, 2] in iso-z
        T_f = p.shape[0]
        p = p.view(T_f, B, 13, 2)
        p_ft = p * eqm_sigma.to(device) + eqm_mu.to(device)
        preds.append(p_ft)
    return torch.stack(preds).mean(0)  # [T_f, B, 13, 2] in feet


def cache_split(name, files, eqm_models, eqm_mu, eqm_sigma, mart_mu, mart_sigma,
                hoops_raw, windows_per_seq, batch_size, device, has_target=True):
    """Stream-process files, batch-run EqMotion ensemble, collect cache."""
    past_all, target_all, eqm_pred_all, ids_all, inv_all = [], [], [], [], []

    # First pass: enumerate all windows
    todo = []  # list of (file, start, raw_seq, canonical_order, canonical_ids, inv_order)
    for f in files:
        raw_seq = torch.load(f, weights_only=False).float()  # [T, 11, 4]
        T = raw_seq.shape[0]
        canonical_order, canonical_ids, inv_order = _canonical_remap(raw_seq)
        if has_target:
            max_start = max(0, T - 20)
            starts = _starts_for(max_start, windows_per_seq)
        else:
            # Test: 8 frames context only, no target. One window at start=0.
            assert T >= 8, f'test seq {f} has T={T} < 8'
            starts = [0]
        for s in starts:
            todo.append((s, raw_seq, canonical_order, canonical_ids, inv_order))

    print(f'  {name}: {len(files)} files -> {len(todo)} windows')

    # Second pass: batch EqMotion forward
    for batch_start in range(0, len(todo), batch_size):
        chunk = todo[batch_start:batch_start + batch_size]
        eqm_inputs = []
        meta = []
        for (s, raw_seq, co, cids, inv) in chunk:
            past = raw_seq[s:s + 8]  # [8, 11, 4]
            eqm_inputs.append(_build_eqm_input(past, eqm_mu, eqm_sigma, hoops_raw))
            meta.append((s, raw_seq, co, cids, inv))
        eqm_input_batch = torch.stack(eqm_inputs)  # [B, 8, 13, 6]
        eqm_pred_ft_batch = run_eqm_ensemble(
            eqm_models, eqm_input_batch, eqm_mu, eqm_sigma, device
        ).cpu()  # [T_f, B, 13, 2]

        for i, (s, raw_seq, co, cids, inv) in enumerate(meta):
            # Past (canonical, MART z-scored): permute to [11, 8, 2]
            past_raw = raw_seq[s:s + 8, co, :2]  # [8, 11, 2] in canonical order
            past_canon = past_raw.permute(1, 0, 2)  # [11, 8, 2]
            past_mart_z = (past_canon - mart_mu) / mart_sigma
            past_all.append(past_mart_z)

            # Target (canonical, MART z-scored)
            if has_target:
                tgt_raw = raw_seq[s + 8:s + 20, co, :2]  # [12, 11, 2]
                tgt_canon = tgt_raw.permute(1, 0, 2)     # [11, 12, 2]
                tgt_mart_z = (tgt_canon - mart_mu) / mart_sigma
                target_all.append(tgt_mart_z)

            # EqMotion pred (drop hoops, reorder to canonical, permute to [11, 12, 2])
            eqm_pred = eqm_pred_ft_batch[:, i, :11]                # [T_f, 11, 2] orig order
            eqm_pred_canon = eqm_pred[:, co, :]                    # [T_f, 11, 2] canonical
            eqm_pred_canon = eqm_pred_canon.permute(1, 0, 2)       # [11, T_f, 2]
            eqm_pred_all.append(eqm_pred_canon)

            ids_all.append(cids)
            inv_all.append(inv)

    cache = {
        'past': torch.stack(past_all),
        'eqm_pred_ft': torch.stack(eqm_pred_all),
        'agent_ids': torch.stack(ids_all),
        'inv_order': torch.stack(inv_all),
        'mart_mu': mart_mu,
        'mart_sigma': mart_sigma,
    }
    if has_target:
        cache['target'] = torch.stack(target_all)
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eqmotion_ckpts', nargs='+', required=True,
                   help='Glob or list of EqMotion .ckpt paths (5-seed ensemble).')
    p.add_argument('--mart_ckpt', required=True,
                   help='Any MART .ckpt — used only to read (mu, sigma).')
    p.add_argument('--split_path', required=True)
    p.add_argument('--test_dir', required=True)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--windows_per_seq', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] device:', device)

    # ---- Load MART (mu, sigma) ----
    mart_ckpt = torch.load(args.mart_ckpt, map_location='cpu', weights_only=False)
    mart_mu = mart_ckpt['mu'].float()
    mart_sigma = mart_ckpt['sigma'].float()
    print(f'[INFO] MART (aniso) mu={mart_mu.tolist()} sigma={mart_sigma.tolist()}')

    # ---- Load EqMotion ensemble + compute its iso stats ----
    # EqMotion needs iso (mu, sigma) — its checkpoints don't save them
    # (computed at runtime via NBADataModule.setup()), so we recompute.
    project_root = Path(args.split_path).resolve().parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    from equivariance.eqmotion_nba import (
        NBADataModule, LANDMARK_SETS, NBAEqMotionLightningModel,
    )

    eqm_dm = NBADataModule(
        split_path=str(args.split_path), batch_size=64, iso_norm=True,
        landmarks=LANDMARK_SETS['hoops'], full_val=False,
    )
    eqm_dm.setup()
    eqm_mu = eqm_dm.mu.float()
    eqm_sigma = eqm_dm.sigma.float()
    print(f'[INFO] EqMotion (iso) mu={eqm_mu.tolist()} sigma={eqm_sigma.tolist()}')

    ckpt_paths = sorted({c for g in args.eqmotion_ckpts for c in glob.glob(g)})
    assert ckpt_paths, f'no checkpoints matched {args.eqmotion_ckpts}'
    print(f'[INFO] loading {len(ckpt_paths)} EqMotion seeds:')
    eqm_models = []
    for cp in ckpt_paths:
        print(f'    {cp}')
        m = NBAEqMotionLightningModel.load_from_checkpoint(
            cp, strict=False, n_landmarks=2,
        )
        m.register_buffer('mu', eqm_mu.clone())
        m.register_buffer('sigma', eqm_sigma.clone())
        eqm_models.append(m.to(device).eval())

    hoops_raw = torch.tensor([[-41.75, 0.0], [41.75, 0.0]], dtype=torch.float32)

    # ---- Cache splits ----
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir, train_files, val_files = _resolve_data_dir(args.split_path)
    train_files = [data_dir / f for f in train_files]
    val_files = [data_dir / f for f in val_files]
    test_files = sorted(Path(args.test_dir).glob('*.pt'),
                        key=lambda p: int(p.stem))
    # remember original test ids so the submission script can write them
    test_ids = [int(p.stem) for p in test_files]

    for name, files, has_target in [
        ('train', train_files, True),
        ('val', val_files, True),
        ('test', test_files, False),
    ]:
        print(f'[INFO] caching {name}')
        c = cache_split(
            name, files, eqm_models, eqm_mu, eqm_sigma, mart_mu, mart_sigma,
            hoops_raw, args.windows_per_seq, args.batch_size, device, has_target,
        )
        if name == 'test':
            c['test_ids'] = torch.tensor(test_ids, dtype=torch.long)
        out = cache_dir / f'{name}.pt'
        torch.save(c, out)
        print(f'  wrote {out}  past:{tuple(c["past"].shape)} '
              f'eqm_pred_ft:{tuple(c["eqm_pred_ft"].shape)}')


if __name__ == '__main__':
    main()
