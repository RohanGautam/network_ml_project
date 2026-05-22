"""
Train NBAHHTCFILightningModel with the best hyperparameters found by Optuna
(see optuna_hht_cfi.db, study 'hht_cfi_optuna', best trial #26).

Full training set, ~200 epochs, early stopping on val/mse_ft.

Run:
    conda run -n nanovlm python src/hht_cfi/train_best.py
"""

import sys
from pathlib import Path

import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

_SRC  = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))

from hht_cfi.hht_cfi_nba import NBADataModule, NBAHHTCFILightningModel  # noqa: E402

# Best hyperparameters from Optuna trial #26 (val/mse_ft = 1.5550 on 25% data)
HPARAMS = dict(
    hidden_size      = 128,
    x_encoder_layers = 3,
    x_encoder_head   = 4,
    lr               = 6.276816713060273e-4,
)
BATCH_SIZE  = 8
GRAD_CLIP   = 2.0
MAX_EPOCHS  = 100
PATIENCE    = 15
RUN_NAME    = "hht_cfi_optuna_best"
SPLIT_PATH  = str(_ROOT / "splits" / "fold0.json")


def main():
    L.seed_everything(0)

    dm    = NBADataModule(split_path=SPLIT_PATH, batch_size=BATCH_SIZE)
    model = NBAHHTCFILightningModel(**HPARAMS)
    print(f"Starting fresh run: {RUN_NAME}")
    print(f"HPARAMS = {HPARAMS}")
    print(f"  batch_size={BATCH_SIZE}  grad_clip={GRAD_CLIP}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}")

    wandb_logger = WandbLogger(
        project   = "NML_base",
        name      = RUN_NAME,
        log_model = False,
        config    = {**HPARAMS, "batch_size": BATCH_SIZE, "grad_clip": GRAD_CLIP},
    )

    log_dir = _ROOT / "logs" / "experiments" / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath    = log_dir,
        filename   = "best-{epoch:03d}-mse{val/mse_ft:.3f}",
        monitor    = "val/mse_ft",
        mode       = "min",
        save_top_k = 1,
    )
    callbacks = [
        EarlyStopping(monitor="val/mse_ft", patience=PATIENCE, mode="min", verbose=True),
        ckpt_cb,
    ]

    trainer = L.Trainer(
        max_epochs        = MAX_EPOCHS,
        accelerator       = "auto",
        gradient_clip_val = GRAD_CLIP,
        log_every_n_steps = 5,
        logger            = wandb_logger,
        callbacks         = callbacks,
    )

    trainer.fit(model, dm)
    wandb.finish()

    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Best val/mse_ft: {ckpt_cb.best_model_score:.4f} ft²")


if __name__ == "__main__":
    main()
