"""Optuna hyperparameter search for dNRI on NBA trajectory data.

Mirrors src/equivariance/tune_eqmotion.py: sequential trials inside a single SLURM
job, TPE sampling, median pruning on val/loss, persistent SQLite study so a
re-launched job resumes where it left off.

Usage (locally or under sbatch):
    python src/dynamic/tune_dnri.py \
        --study-path /home/rgautam/network_ml_project/optuna_studies/dnri.db \
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

from equivariance.eqmotion_nba import NBADataModule  # noqa: E402
from dynamic.dnri_nba import NBADNRILightningModel  # noqa: E402


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
        # Capacity
        encoder_hidden = trial.suggest_categorical("encoder_hidden", [64, 128, 256])
        encoder_rnn_hidden = trial.suggest_categorical("encoder_rnn_hidden", [32, 64, 128])
        decoder_hidden = trial.suggest_categorical("decoder_hidden", [32, 64, 128])
        # MLP heads sized off the RNN output — keep them proportionate.
        encoder_mlp_hidden = max(32, encoder_rnn_hidden)
        prior_hidden_size = max(32, encoder_rnn_hidden)

        # dNRI-specific
        num_edge_types = trial.suggest_int("num_edge_types", 2, 4)
        gumbel_temp = trial.suggest_float("gumbel_temp", 0.1, 1.0, log=True)
        kl_coef = trial.suggest_float("kl_coef", 1e-2, 1.0, log=True)
        encoder_dropout = trial.suggest_float("encoder_dropout", 0.0, 0.3)
        decoder_dropout = trial.suggest_float("decoder_dropout", 0.0, 0.3)

        # Optimization
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
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

        model = NBADNRILightningModel(
            context_size=8,
            horizon_size=12,
            num_edge_types=num_edge_types,
            lr=lr,
            weight_decay=weight_decay,
            # dNRI model overrides (forwarded to default_dnri_params)
            encoder_hidden=encoder_hidden,
            encoder_rnn_hidden=encoder_rnn_hidden,
            decoder_hidden=decoder_hidden,
            encoder_mlp_hidden=encoder_mlp_hidden,
            prior_hidden_size=prior_hidden_size,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            gumbel_temp=gumbel_temp,
            kl_coef=kl_coef,
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
    parser.add_argument("--study-name", default="dnri_v1")
    parser.add_argument(
        "--study-path",
        default=str(PROJECT_ROOT / "optuna_studies" / "dnri.db"),
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
