"""Download val/mse_ft training curves from W&B for the report figures.

The W&B CSV export silently drops runs (only the two EqMotion runs came through),
so we pull the full history per run directly via the public API instead.

Usage:
    uv run src/report/fetch_wandb_curves.py

Reads WANDB_API_KEY (and optionally WANDB_ENTITY) from the project .env.
Writes one tidy CSV to src/report/wandb_val_mse_ft.csv with columns:
    run, step, epoch, val_mse_ft
(`epoch` gives a common x-axis across runs that log validation at different
internal step cadences.)
"""

import os
from pathlib import Path

import dotenv
import pandas as pd
import wandb

PROJECT = "NML_base"
METRIC = "val/mse_ft"

# Runs shown in the W&B chart for this figure (display names).
RUNS = [
    "eqm_iso_hoops_5k",            # EqMotion, final 5k run
    "mart_aug_iso_hoops_5k",       # MART trained with min-ADE loss
    "mart_meanmse_iso_hoops_5k",   # MART trained with mean-MSE loss
    "mart_curriculum_4k_sw1k_lrdrop",  # MART curriculum (min-ADE -> mean-MSE), best
    "iso_hoops_bw5",               # shorter EqMotion (ball-weight 5) — kept for completeness
]

OUT = Path(__file__).resolve().parent / "wandb_val_mse_ft.csv"


def main() -> None:
    dotenv.load_dotenv(dotenv.find_dotenv())
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY not found in environment / .env")
    wandb.login(key=api_key)

    api = wandb.Api()
    entity = os.getenv("WANDB_ENTITY") or api.default_entity
    print(f"entity={entity}  project={PROJECT}")

    # Map display name -> run object (a run's `name` is the id; `display_name` is the label).
    runs = list(api.runs(f"{entity}/{PROJECT}"))
    by_display = {r.display_name: r for r in runs}

    frames = []
    for name in RUNS:
        run = by_display.get(name)
        if run is None:
            print(f"  [skip] '{name}' not found in {entity}/{PROJECT}")
            continue
        # Pull the metric plus `epoch` (Lightning logs it) for a common x-axis.
        hist = run.history(keys=[METRIC, "epoch"], pandas=True)
        if hist.empty or METRIC not in hist:
            print(f"  [warn] '{name}' has no '{METRIC}' history")
            continue
        cols = {"_step": "step", METRIC: "val_mse_ft"}
        keep = ["_step", METRIC]
        if "epoch" in hist:
            keep.append("epoch")
        df = hist[keep].dropna(subset=[METRIC]).rename(columns=cols)
        if "epoch" not in df:
            df["epoch"] = range(len(df))  # fallback: ordinal validation index
        df.insert(0, "run", name)
        frames.append(df[["run", "step", "epoch", "val_mse_ft"]])
        print(f"  [ok]   {name}: {len(df)} points, "
              f"epoch {df.epoch.min():.0f}..{df.epoch.max():.0f}, "
              f"min={df.val_mse_ft.min():.3f} final={df.val_mse_ft.iloc[-1]:.3f}")

    if not frames:
        raise SystemExit("No run histories downloaded.")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
