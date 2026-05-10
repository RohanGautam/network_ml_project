"""Optuna hyperparameter search for EqMotion on NBA trajectory data.

Sequential trials inside a single SLURM job, with TPE sampling and median pruning
on val/loss. Study state persists in SQLite so a re-launched job resumes where it
left off.

Usage (locally or under sbatch):
    python src/equivariance/tune_eqmotion.py \
        --study-path /home/rgautam/network_ml_project/optuna_studies/eqmotion.db \
        --n-trials 40 --max-epochs 50 --timeout 16200
"""

import argparse
import sys
from pathlib import Path

import lightning as L
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equivariance.eqmotion_nba import (  # noqa: E402
    NBADataModule,
    NBAEqMotionLightningModel,
)


class OptunaPruning(L.Callback):
    """Report val/loss to Optuna after each validation epoch and prune if asked."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val/loss"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value.item()), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def build_objective(args):
    split_path = str(PROJECT_ROOT / "splits" / "fold0.json")

    def objective(trial: optuna.Trial) -> float:
        hidden_nf = trial.suggest_categorical("hidden_nf", [32, 64, 128, 256])
        hid_channel = trial.suggest_categorical("hid_channel", [8, 16, 32, 64])
        n_layers = trial.suggest_int("n_layers", 2, 5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        grad_clip = trial.suggest_float("gradient_clip_val", 0.25, 2.0)

        L.seed_everything(args.seed, workers=True)

        data_module = NBADataModule(
            split_path=split_path,
            batch_size=batch_size,
            context_size=8,
            horizon_size=12,
            seed=args.seed,
        )

        model = NBAEqMotionLightningModel(
            context_size=8,
            horizon_size=12,
            hidden_nf=hidden_nf,
            hid_channel=hid_channel,
            n_layers=n_layers,
            lr=lr,
            weight_decay=weight_decay,
        )

        wandb_logger = WandbLogger(
            project="NML_tune",
            name=f"trial_{trial.number}",
            group=args.study_name,
            config=trial.params,
            reinit=True,
        )

        callbacks = [
            EarlyStopping(monitor="val/loss", patience=args.patience, mode="min"),
            OptunaPruning(trial, monitor="val/loss"),
        ]

        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            logger=wandb_logger,
            accelerator="auto",
            gradient_clip_val=grad_clip,
            callbacks=callbacks,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=20,
        )

        try:
            trainer.fit(model, data_module)
            metric = trainer.callback_metrics.get("val/loss")
            if metric is None:
                raise optuna.TrialPruned()
            return float(metric.item())
        finally:
            wandb.finish()
            torch.cuda.empty_cache()

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="eqmotion_v1")
    parser.add_argument(
        "--study-path",
        default=str(PROJECT_ROOT / "optuna_studies" / "eqmotion.db"),
        help="Path to SQLite file backing the Optuna study (persistent across jobs).",
    )
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-clock seconds budget for study.optimize. Set below SLURM --time.",
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=8,
        help="Random trials before TPE/pruner kick in.",
    )
    args = parser.parse_args()

    study_path = Path(args.study_path)
    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_path}"

    sampler = TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials)
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=10,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(
        build_objective(args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best val/loss: {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
