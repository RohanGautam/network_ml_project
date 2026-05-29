"""Generate a Kaggle submission CSV from a trained EqMotion checkpoint.

Critical: the datamodule must be rebuilt with the SAME normalization used during
training (e.g. --iso-norm for the iso_sched checkpoint), otherwise denormalization
of predictions back to feet is wrong and the submission is garbage.

Usage:
    python src/equivariance/submit_eqmotion.py \
        --ckpt checkpoints/eqmotion/iso_sched/best.ckpt --iso-norm
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equivariance.eqmotion_nba import (  # noqa: E402
    SUBMISSION_DIR,
    TEST_DIR,
    NBADataModule,
    NBAEqMotionLightningModel,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--iso-norm", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    dm = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=args.batch_size,
        iso_norm=args.iso_norm,
    )
    dm.setup()  # computes mu/sigma from train split with matching normalization
    print(f"normalization: iso={args.iso_norm}  mu={dm.mu.tolist()}  sigma={dm.sigma.tolist()}")

    # mu/sigma are buffers only registered in on_fit_start; we pass them explicitly
    # to get_trajectory, so dropping the (absent) keys on load is safe.
    model = NBAEqMotionLightningModel.load_from_checkpoint(args.ckpt, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    dm.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR))
    print("submission written to", SUBMISSION_DIR)


if __name__ == "__main__":
    main()
