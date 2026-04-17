"""Deterministic train/val split manifest for the NBA dataset.

Usage:
    python -m src.utils.split_dataset --seed 0 --val-frac 0.1 --name fold0

Writes the manifest to `splits/<name>.json` at the project root. Everyone on the
team generates the same split from the same seed, so only the tiny JSON needs
to be committed — the .pt files stay wherever they already live.
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "train"
SPLITS_DIR = PROJECT_ROOT / "splits"


def make_split(data_dir: Path, seed: int, val_frac: float) -> dict:
    files = sorted(f.name for f in data_dir.iterdir() if f.suffix == ".pt")
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = round(len(files) * val_frac)
    return {
        "seed": seed,
        "val_frac": val_frac,
        "data_dir": str(data_dir.relative_to(PROJECT_ROOT)),
        "train": sorted(files[n_val:]),
        "val": sorted(files[:n_val]),
    }


def load_split(manifest_path: Path) -> tuple[list[Path], list[Path]]:
    """Resolve a manifest into absolute file paths for train and val."""
    manifest = json.loads(manifest_path.read_text())
    data_dir = PROJECT_ROOT / manifest["data_dir"]
    train = [data_dir / f for f in manifest["train"]]
    val = [data_dir / f for f in manifest["val"]]
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--name", type=str, default="fold0")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    args = parser.parse_args()

    manifest = make_split(args.data_dir, args.seed, args.val_frac)
    SPLITS_DIR.mkdir(exist_ok=True)
    out = SPLITS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}  (train={len(manifest['train'])}, val={len(manifest['val'])})")


if __name__ == "__main__":
    main()
