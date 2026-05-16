"""Retrain the top-K Optuna trials at full epoch budget and emit two Kaggle
submissions: (1) the single best model, (2) the top-K mean ensemble.

The Optuna study at --study-path is opened read-only; trials are filtered to
COMPLETE state and ranked by val/loss. Each rank is retrained from scratch with
its trial params (no warm-start), checkpointed on best val/loss, then used to
predict the test set. Predictions are stacked across ranks; rank-0 alone yields
the "best" CSV and the per-coordinate mean yields the "ensemble" CSV.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import optuna
import pandas as pd
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equivariance.eqmotion_nba import (  # noqa: E402
    SUBMISSION_DIR,
    TEST_DIR,
    NBADataModule,
    NBAEqMotionLightningModel,
)


def predict_test(model, dm, test_dir):
    rows, ids = [], []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".pt"):
            continue
        seq = torch.load(os.path.join(test_dir, f), weights_only=False)
        seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - dm.mu) / dm.sigma
        vel = torch.zeros_like(seq[:, :, :2])
        vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
        seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)
        traj = model.get_trajectory(seq, dm.mu, dm.sigma)
        rows.append(traj[8:, :, :2].reshape(-1).numpy())
        ids.append(int(f.removesuffix(".pt")))
    return np.asarray(ids), np.stack(rows)


def write_submission(ids, preds, out_path):
    cols = [
        f"entity_{i}_time_{t}_{axis}"
        for t in range(12)
        for i in range(11)
        for axis in ["x", "y"]
    ]
    df = pd.DataFrame(preds, columns=cols)
    df.insert(0, "id", ids)
    df.set_index("id").sort_index().to_csv(out_path)


def train_one(trial, rank, args, ckpt_dir):
    p = trial.params
    L.seed_everything(args.seed, workers=True)

    dm = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=p["batch_size"],
        context_size=8,
        horizon_size=12,
        seed=args.seed,
    )

    model = NBAEqMotionLightningModel(
        context_size=8,
        horizon_size=12,
        hidden_nf=p["hidden_nf"],
        hid_channel=p["hid_channel"],
        n_layers=p["n_layers"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"rank{rank}_trial{trial.number}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    wandb_logger = WandbLogger(
        project="NML_topk",
        name=f"rank{rank}_trial{trial.number}",
        group=args.run_name,
        config=p,
        reinit=True,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        accelerator="auto",
        gradient_clip_val=p["gradient_clip_val"],
        callbacks=[ckpt_cb, early_stop],
    )
    trainer.fit(model, dm)

    # strict=False: mu/sigma are registered as buffers only in on_fit_start, so
    # they live in the state_dict but not on a freshly-built model. We don't use
    # self.mu/self.sigma at predict time (get_trajectory takes them as args), so
    # silently dropping these keys is safe.
    best = NBAEqMotionLightningModel.load_from_checkpoint(
        ckpt_cb.best_model_path, strict=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = best.to(device)
    best.eval()

    best_val = float(ckpt_cb.best_model_score.item())
    print(f"[rank {rank}] trial {trial.number}: best val/loss = {best_val:.6f}")

    ids, preds = predict_test(best, dm, str(TEST_DIR))

    wandb.finish()
    del best, model, trainer
    torch.cuda.empty_cache()

    return ids, preds, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-path",
        default=str(PROJECT_ROOT / "optuna_studies" / "eqmotion.db"),
    )
    parser.add_argument("--study-name", default="eqmotion_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="topk_retrain")
    args = parser.parse_args()

    storage = f"sqlite:///{args.study_path}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    top = completed[: args.top_k]

    print(f"Loaded {len(completed)} completed trials. Retraining top {len(top)}:")
    for rank, t in enumerate(top):
        print(f"  rank {rank}: trial {t.number}, val/loss(tuning) = {t.value:.6f}")
        print(f"           params = {t.params}")

    ckpt_dir = PROJECT_ROOT / "models" / "topk"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_preds, common_ids, val_losses = [], None, []
    for rank, trial in enumerate(top):
        ids, preds, val_loss = train_one(trial, rank, args, str(ckpt_dir))
        if common_ids is None:
            common_ids = ids
        else:
            assert (ids == common_ids).all(), "test id ordering changed between runs"
        all_preds.append(preds)
        val_losses.append(val_loss)

    stacked = np.stack(all_preds)  # [K, n_test, 264]
    SUBMISSION_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_path = SUBMISSION_DIR / f"solution_best_{timestamp}.csv"
    write_submission(common_ids, stacked[0], best_path)

    ensemble_path = SUBMISSION_DIR / f"solution_ensemble_{timestamp}.csv"
    write_submission(common_ids, stacked.mean(axis=0), ensemble_path)

    print("\n=== Retrain summary ===")
    for rank, (trial, vl) in enumerate(zip(top, val_losses)):
        print(f"  rank {rank}: trial {trial.number}, retrained val/loss = {vl:.6f}")
    print(f"\nBest-model submission:  {best_path}")
    print(f"Ensemble submission:    {ensemble_path}")


if __name__ == "__main__":
    main()
