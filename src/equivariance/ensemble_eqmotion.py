"""Multi-seed ensemble (+ optional reflection TTA) for EqMotion iso+hoops.

Evaluates single-model vs ensemble, each with/without 4-way court-symmetry
reflection TTA, on the deterministic full-val split using the HONEST 11-entity
metric (hoops stripped). Then writes Kaggle submissions for the ensemble.

Note: EqMotion is O(2)-equivariant, so reflection TTA is expected to be ~a no-op
(it confirms equivariance rather than improving). The ensemble is the real lever.

Usage:
    python src/equivariance/ensemble_eqmotion.py \
        --ckpts checkpoints/eqmotion/iso_hoops_s*/best.ckpt
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equivariance.eqmotion_nba import (  # noqa: E402
    SUBMISSION_DIR,
    TEST_DIR,
    NBADataModule,
    NBAEqMotionLightningModel,
)

N_ENTITIES = 11  # players + ball; hoops (if present) are appended after these
HORIZON = 12
FLIPS = [(), (0,), (1,), (0, 1)]  # court-symmetry reflections (D2)


def _fwd(model, X):
    """X [B,T,N,6] normalized -> pred [T_f, B, N, 2] normalized."""
    p = model(X)  # [T_f, B*N, 2]
    return p.view(p.shape[0], X.shape[0], X.shape[2], 2)


def _fwd_tta(model, X):
    """Average predictions over the 4 court-symmetry reflections."""
    acc = None
    for axes in FLIPS:
        Xf = X.clone()
        for a in axes:
            Xf[..., a] = -Xf[..., a]       # position channel
            Xf[..., a + 2] = -Xf[..., a + 2]  # velocity channel
        p = _fwd(model, Xf)
        for a in axes:
            p[..., a] = -p[..., a]         # un-reflect output
        acc = p if acc is None else acc + p
    return acc / len(FLIPS)


def predict(models, X, tta):
    """Mean normalized prediction across models (+ optional TTA). -> [T_f,B,N,2]."""
    fn = _fwd_tta if tta else _fwd
    return torch.stack([fn(m, X) for m in models]).mean(0)


@torch.no_grad()
def eval_val(models, dm, device):
    """Report honest 11-entity val/mse_ft for {single,ensemble}x{plain,tta}."""
    mu, sigma = dm.mu.to(device), dm.sigma.to(device)
    configs = {"single_plain": ([models[0]], False),
               "single_tta": ([models[0]], True),
               "ensemble_plain": (models, False),
               "ensemble_tta": (models, True)}
    sse = {k: 0.0 for k in configs}
    cnt = {k: 0 for k in configs}
    for X, y in dm.val_dataloader():
        X = X.to(device)
        B, T, N, _ = y.shape
        tgt = y[..., :2].permute(1, 0, 2, 3).to(device) * sigma + mu  # [T,B,N,2]
        tgt = tgt[:, :, :N_ENTITIES, :]
        for k, (ms, tta) in configs.items():
            pred = predict(ms, X, tta) * sigma + mu
            pred = pred[:, :, :N_ENTITIES, :]
            sse[k] += ((pred - tgt) ** 2).sum().item()
            cnt[k] += pred.numel()
    return {k: sse[k] / cnt[k] for k in configs}


def _build_test_X(seq, mu, sigma, add_hoops):
    """Test seq [8,11,4] -> normalized X [1, 8, N, 6]."""
    pos = (seq[:, :, :2] - mu) / sigma
    vel = torch.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    X = torch.cat([pos, vel, seq[:, :, 2:]], dim=-1)  # [8,11,6]
    if add_hoops:
        T = X.shape[0]
        raw = torch.tensor([[-41.75, 0.0], [41.75, 0.0]])
        nh = (raw - mu) / sigma
        hoop = torch.zeros((T, 2, 6), dtype=X.dtype)
        hoop[:, :, :2] = nh
        hoop[:, :, 5] = 3.0  # team id for landmarks
        X = torch.cat([X, hoop], dim=1)  # [8,13,6]
    return X.unsqueeze(0)


@torch.no_grad()
def write_submission(models, dm, device, tta, tag):
    mu, sigma = dm.mu.to(device), dm.sigma.to(device)
    rows, ids = [], []
    for f in sorted(os.listdir(TEST_DIR)):
        if not f.endswith(".pt"):
            continue
        seq = torch.load(os.path.join(TEST_DIR, f), weights_only=False).float()
        X = _build_test_X(seq, dm.mu, dm.sigma, dm.add_hoops).to(device)
        pred = predict(models, X, tta)[:, 0] * sigma + mu  # [T_f, N, 2]
        traj = pred[:, :N_ENTITIES, :2].reshape(-1).cpu().numpy()
        rows.append(traj)
        ids.append(int(f.removesuffix(".pt")))
    cols = [f"entity_{i}_time_{t}_{ax}" for t in range(HORIZON)
            for i in range(N_ENTITIES) for ax in ["x", "y"]]
    df = pd.DataFrame(np.stack(rows), columns=cols)
    df.insert(0, "id", ids)
    df = df.set_index("id").sort_index()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = SUBMISSION_DIR / f"solution_ens_{tag}_{ts}.csv"
    df.to_csv(out)
    print(f"  wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="Checkpoint paths or globs.")
    ap.add_argument("--iso-norm", action="store_true", default=True)
    ap.add_argument("--add-hoops", action="store_true", default=True)
    args = ap.parse_args()

    paths = sorted({p for g in args.ckpts for p in glob.glob(g)})
    assert paths, f"no checkpoints matched {args.ckpts}"
    print(f"Ensembling {len(paths)} checkpoints:")
    for p in paths:
        print("  ", p)

    dm = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=64, iso_norm=args.iso_norm, add_hoops=args.add_hoops,
        full_val=True,
    )
    dm.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_land = 2 if args.add_hoops else 0
    models = []
    for p in paths:
        m = NBAEqMotionLightningModel.load_from_checkpoint(p, strict=False, n_landmarks=n_land)
        m.register_buffer("mu", dm.mu.clone())
        m.register_buffer("sigma", dm.sigma.clone())
        models.append(m.to(device).eval())

    print("\n=== val/mse_ft (honest, 11 entities) ===")
    res = eval_val(models, dm, device)
    for k, v in res.items():
        print(f"  {k:16s} {v:.4f}")

    print("\n=== submissions ===")
    write_submission(models, dm, device, tta=False, tag="plain")
    write_submission(models, dm, device, tta=True, tag="tta")


if __name__ == "__main__":
    main()
