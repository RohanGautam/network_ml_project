"""Optuna hyperparameter tuning for the Social-STGCNN pipeline, with top-K
retrain + ensemble in the same script.

Two phases in one run (either can be skipped with --no-tune / --no-retrain):

  1. TUNE   — an Optuna study (persistent sqlite, resumable across job relaunches)
              minimizes val/mse_ft over short training runs with median pruning.
  2. RETRAIN— the top-K completed trials are retrained from scratch at full
              epoch budget, each predicts the test set (with court-symmetry TTA),
              and we emit two Kaggle submissions: the single best model and the
              top-K per-coordinate mean ensemble (cf. equivariance/retrain_topk.py).

Everything reuses the model/data/submission code in stgcnn_nba.py; tuning only
searches the architecture/optim knobs exposed by NBASTGCNNLightningModel. The
objective (and submission) is val/mse_ft, i.e. the Kaggle metric.
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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgcnn.stgcnn_nba import (  # noqa: E402
    N_ENTITIES,
    SUBMISSION_DIR,
    TEST_DIR,
    NBADataModule,
    NBASTGCNNLightningModel,
)

SPLIT_PATH = str(PROJECT_ROOT / "splits" / "fold0.json")


# ── Search space + builders ─────────────────────────────────────────────────


def suggest_params(trial: optuna.Trial) -> dict:
    """Architecture / optimization knobs. loss_mode and optimizer are fixed to
    the configuration we already established as best (MSE-on-metric + Adam)."""
    return {
        "graph_space": trial.suggest_categorical("graph_space", ["pos", "vel", "both"]),
        "decoder": trial.suggest_categorical("decoder", ["autoreg", "txp"]),
        "n_stgcnn": trial.suggest_int("n_stgcnn", 1, 3),
        "hidden_feat": trial.suggest_categorical("hidden_feat", [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "n_txpcnn": trial.suggest_int("n_txpcnn", 3, 7, step=2),  # used only by "txp"
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "augment": trial.suggest_categorical("augment", [True, False]),
    }


def build(p: dict, seed: int):
    dm = NBADataModule(
        split_path=SPLIT_PATH,
        batch_size=p["batch_size"],
        seed=seed,
        graph_space=p["graph_space"],
    )
    model = NBASTGCNNLightningModel(
        loss_mode="mse",
        optimizer="adam",
        graph_space=p["graph_space"],
        decoder=p["decoder"],
        n_stgcnn=p["n_stgcnn"],
        hidden_feat=p["hidden_feat"],
        kernel_size=p["kernel_size"],
        n_txpcnn=p["n_txpcnn"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        augment=p["augment"],
    )
    return dm, model


class OptunaPruning(L.Callback):
    """Report val metric to the trial each epoch and prune unpromising trials."""

    def __init__(self, trial, monitor="val/mse_ft"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get(self.monitor)
        if val is None or trainer.sanity_checking:
            return
        self.trial.report(val.item(), trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ── Objective ────────────────────────────────────────────────────────────────


def objective(trial: optuna.Trial, args) -> float:
    p = suggest_params(trial)
    L.seed_everything(args.seed, workers=True)
    dm, model = build(p, args.seed)

    ckpt = ModelCheckpoint(
        dirpath=str(PROJECT_ROOT / "optuna_studies" / "trial_ckpts" / f"t{trial.number}"),
        filename="best", monitor="val/mse_ft", mode="min", save_top_k=1,
    )
    callbacks = [
        ckpt,
        EarlyStopping(monitor="val/mse_ft", mode="min", patience=args.patience),
        OptunaPruning(trial, "val/mse_ft"),
    ]
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, dm)
    best = ckpt.best_model_score
    return float(best.item()) if best is not None else float("inf")


# ── Test prediction + submission (mirrors stgcnn_nba.get_kaggle_submission) ────


def predict_test(model, dm, test_dir, tta=True):
    """Return (ids, preds[n_test, 264]) for one model, with court-symmetry TTA."""
    c = dm.context_size
    flips = [(), (0,), (1,), (0, 1)] if tta else [()]
    ids, rows = [], []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".pt"):
            continue
        seq = torch.load(os.path.join(test_dir, f), weights_only=False).float()
        abs_obs = seq[:, :, :2]
        static = seq[c - 1, :, 2:4]
        preds = []
        for axes in flips:
            ao = abs_obs.clone()
            for ax in axes:
                ao[..., ax] = -ao[..., ax]
            pr = dm._predict_abs(model, ao, static)  # [pred, N, 2]
            for ax in axes:
                pr[..., ax] = -pr[..., ax]
            preds.append(pr)
        ap = torch.stack(preds).mean(0)  # [pred, N, 2]
        rows.append(ap[:, :N_ENTITIES, :2].reshape(-1).numpy())
        ids.append(int(f.removesuffix(".pt")))
    return np.asarray(ids), np.stack(rows)


def write_submission(ids, preds, out_path):
    cols = [
        f"entity_{i}_time_{t}_{axis}"
        for t in range(12)
        for i in range(N_ENTITIES)
        for axis in ["x", "y"]
    ]
    df = pd.DataFrame(preds, columns=cols)
    df.insert(0, "id", ids)
    df.set_index("id").sort_index().to_csv(out_path)


def retrain_one(trial, rank, args):
    p = trial.params
    L.seed_everything(args.seed, workers=True)
    dm, model = build(p, args.seed)

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "stgcnn_topk"
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename=f"rank{rank}_trial{trial.number}",
        monitor="val/mse_ft", mode="min", save_top_k=1,
    )
    callbacks = [ckpt, EarlyStopping(monitor="val/mse_ft", mode="min", patience=args.patience)]
    logger = (
        WandbLogger(project="NML_topk", name=f"stgcnn_rank{rank}_trial{trial.number}",
                    group=args.run_name, config=p, reinit=True)
        if args.wandb else False
    )
    trainer = L.Trainer(
        max_epochs=args.retrain_epochs,
        logger=logger,
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        enable_progress_bar=False,
    )
    trainer.fit(model, dm)

    best = NBASTGCNNLightningModel.load_from_checkpoint(ckpt.best_model_path)
    best = best.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    best_val = float(ckpt.best_model_score.item())
    print(f"[rank {rank}] trial {trial.number}: retrained val/mse_ft = {best_val:.4f}")

    ids, preds = predict_test(best, dm, str(TEST_DIR), tta=not args.no_tta)
    if args.wandb:
        import wandb
        wandb.finish()
    del best, model, trainer
    torch.cuda.empty_cache()
    return ids, preds, best_val


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="stgcnn_v1")
    parser.add_argument("--study-path", default=str(PROJECT_ROOT / "optuna_studies" / "stgcnn.db"))
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=50, help="per-trial budget (tuning)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=None, help="seconds for the tuning phase")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrain-epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="stgcnn_topk")
    parser.add_argument("--no-tune", action="store_true", help="skip tuning, retrain from existing study")
    parser.add_argument("--no-retrain", action="store_true", help="tune only, no retrain/submit")
    parser.add_argument("--no-tta", action="store_true", help="disable TTA at prediction time")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    Path(args.study_path).parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.study_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    # ── Phase 1: tune ──
    if not args.no_tune:
        study.optimize(
            lambda t: objective(t, args),
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    print(f"\n{len(completed)} completed trials. Best val/mse_ft = "
          f"{completed[0].value:.4f}" if completed else "no completed trials")

    if args.no_retrain or not completed:
        return

    # ── Phase 2: retrain top-K + ensemble ──
    top = completed[: args.top_k]
    print(f"\nRetraining top {len(top)} at {args.retrain_epochs} epochs:")
    for rank, t in enumerate(top):
        print(f"  rank {rank}: trial {t.number}, tuning val/mse_ft = {t.value:.4f}  {t.params}")

    all_preds, common_ids, val_losses = [], None, []
    for rank, trial in enumerate(top):
        ids, preds, vl = retrain_one(trial, rank, args)
        if common_ids is None:
            common_ids = ids
        else:
            assert (ids == common_ids).all(), "test id ordering changed between runs"
        all_preds.append(preds)
        val_losses.append(vl)

    stacked = np.stack(all_preds)  # [K, n_test, 264]
    SUBMISSION_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = SUBMISSION_DIR / f"solution_stgcnn_best_{ts}.csv"
    ens_path = SUBMISSION_DIR / f"solution_stgcnn_ensemble_{ts}.csv"
    write_submission(common_ids, stacked[0], best_path)
    write_submission(common_ids, stacked.mean(axis=0), ens_path)

    print("\n=== Retrain summary ===")
    for rank, (trial, vl) in enumerate(zip(top, val_losses)):
        print(f"  rank {rank}: trial {trial.number}, retrained val/mse_ft = {vl:.4f}")
    print(f"\nBest-model submission: {best_path}")
    print(f"Ensemble submission:   {ens_path}")


if __name__ == "__main__":
    main()
