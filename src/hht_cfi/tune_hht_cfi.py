"""
Optuna hyperparameter search for NBAHHTCFILightningModel.

Budget:  ~10 hours total, ~15 min per trial average → 40 trials.
Data:    25% of training files (random subset, fixed seed per trial).
Val set: always full — reliable metric for Optuna's objective.

Run:
    conda run -n nanovlm python src/hht_cfi/tune_hht_cfi.py

Results are persisted to optuna_hht_cfi.db (SQLite), so you can safely
interrupt and resume. To inspect results:
    conda run -n nanovlm python src/hht_cfi/tune_hht_cfi.py --report
"""

import sys
import argparse
from pathlib import Path

import optuna
import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

_SRC  = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))

from hht_cfi.hht_cfi_nba import NBADataModule, NBAHHTCFILightningModel  # noqa: E402

DATA_FRACTION = 0.25   # fraction of training files per trial
MAX_EPOCHS    = 25     # per trial; early stopping will cut most short
PATIENCE      = 7      # early stopping patience on val/mse_ft
N_TRIALS      = 40
STUDY_NAME    = "hht_cfi_optuna"
DB_PATH       = _ROOT / "optuna_hht_cfi.db"
SPLIT_PATH    = str(_ROOT / "splits" / "fold0.json")


class _PruneCallback(L.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val/mse_ft"):
        self.trial   = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(value.item(), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial: optuna.Trial) -> float:
    hidden_size      = trial.suggest_categorical("hidden_size",      [32, 64, 128])
    x_encoder_layers = trial.suggest_int(        "x_encoder_layers", 2, 5)
    # n_heads must divide hidden_size; 4 and 8 both divide 32/64/128
    x_encoder_head   = trial.suggest_categorical("x_encoder_head",   [4, 8])
    lr               = trial.suggest_float(      "lr",               1e-4, 1e-3, log=True)
    batch_size       = trial.suggest_categorical("batch_size",        [8, 16, 32])
    grad_clip        = trial.suggest_categorical("grad_clip",         [0.5, 1.0, 2.0])

    dm = NBADataModule(
        split_path    = SPLIT_PATH,
        batch_size    = batch_size,
        data_fraction = DATA_FRACTION,
        seed          = trial.number,   # different random subset per trial
    )

    L.seed_everything(0)
    model = NBAHHTCFILightningModel(
        hidden_size      = hidden_size,
        x_encoder_layers = x_encoder_layers,
        x_encoder_head   = x_encoder_head,
        lr               = lr,
    )

    wandb_logger = WandbLogger(
        project  = "NML_base",
        name     = f"optuna_trial_{trial.number:03d}",
        log_model= False,
        config   = {
            "hidden_size":      hidden_size,
            "x_encoder_layers": x_encoder_layers,
            "x_encoder_head":   x_encoder_head,
            "lr":               lr,
            "batch_size":       batch_size,
            "grad_clip":        grad_clip,
            "data_fraction":    DATA_FRACTION,
        },
    )

    trainer = L.Trainer(
        max_epochs        = MAX_EPOCHS,
        accelerator       = "auto",
        gradient_clip_val = grad_clip,
        log_every_n_steps = 5,
        enable_checkpointing = False,
        enable_progress_bar  = False,
        logger               = wandb_logger,
        callbacks = [
            EarlyStopping(monitor="val/mse_ft", patience=PATIENCE, mode="min"),
            _PruneCallback(trial, monitor="val/mse_ft"),
        ],
    )

    trainer.fit(model, dm)
    wandb.finish()

    val_mse = trainer.callback_metrics.get("val/mse_ft")
    if val_mse is None:
        raise optuna.TrialPruned()
    return val_mse.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="Print best trial summary and top-10 results, then exit.")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    storage = f"sqlite:///{DB_PATH}"

    if args.report:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
        _print_report(study)
        return

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 5,   # don't prune the first 5 trials
        n_warmup_steps   = 4,   # don't prune before epoch 4
    )
    study = optuna.create_study(
        direction  = "minimize",
        study_name = STUDY_NAME,
        storage    = storage,
        load_if_exists = True,
        pruner     = pruner,
    )

    print(f"Starting Optuna study '{STUDY_NAME}' — {args.n_trials} trials")
    print(f"  Data fraction : {DATA_FRACTION*100:.0f}% of training files per trial")
    print(f"  Max epochs    : {MAX_EPOCHS}  (early stopping patience={PATIENCE})")
    print(f"  DB            : {DB_PATH}\n")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    _print_report(study)


def _print_report(study: optuna.Study):
    print(f"\nOPTUNA RESULTS - {study.study_name}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials   : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    best = study.best_trial
    print(f"\nBest trial #{best.number}  val/mse_ft = {best.value:.4f} ft²")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v}")

    print(f"\nTop-10 completed trials:")
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )
    print(f"  {'#':>4}  {'val/mse_ft':>12}  params")  # noqa
    for t in completed[:10]:
        param_str = "  ".join(f"{k}={v}" for k, v in t.params.items())
        print(f"  {t.number:>4}  {t.value:>12.4f}  {param_str}")

    print(f"\nTo retrain best config, use in notebook Cell 17:")
    print(f"  HPARAMS = {best.params}")


if __name__ == "__main__":
    main()
