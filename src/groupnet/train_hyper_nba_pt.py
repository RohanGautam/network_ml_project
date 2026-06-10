"""Train GroupNet (with learned entity-ID embedding) on network_ml_project data.

Mirrors the original GroupNet trainer (train_hyper_nba.py), but:
    - reads .pt sequences via the network_ml_project split manifest
    - z-scores (x, y) using stats computed from the train split
    - uses past=8, future=12 (Kaggle horizon)
    - feeds agent_ids into GroupNetWithID's nn.Embedding

Example:
    python train_hyper_nba_pt.py \
        --split_path ../network_ml_project/splits/fold0.json \
        --gpu 0
"""

import argparse
import os
import random
import sys
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

import numpy as np
import torch
import wandb
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from data.dataloader_nba_pt import (
    GroupNetNBAPTDataset,
    WindowSampler,
    compute_xy_stats,
    groupnet_collate,
    load_split_files,
)
from model.GroupNet_nba_id import GroupNetWithID


def parse_args():
    p = argparse.ArgumentParser()
    # Data / split
    p.add_argument(
        "--split_path",
        type=str,
        required=True,
        help="Path to network_ml_project splits/<name>.json",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    # Window
    p.add_argument("--past_length", type=int, default=8)
    p.add_argument("--future_length", type=int, default=12)
    # Optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--decay_step", type=int, default=10)
    p.add_argument("--decay_gamma", type=float, default=0.5)
    p.add_argument("--iternum_print", type=int, default=100)
    # GroupNet hyper-params (kept identical to the upstream defaults)
    p.add_argument("--ztype", default="gaussian")
    p.add_argument("--zdim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--hyper_scales", nargs="+", type=int, default=[5, 11])
    p.add_argument("--num_decompose", type=int, default=2)
    p.add_argument("--min_clip", type=float, default=2.0)
    p.add_argument("--sample_k", type=int, default=20)
    p.add_argument("--learn_prior", action="store_true", default=False)
    p.add_argument("--traj_scale", type=int, default=1)  # unused; kept for ckpt compat
    # Entity embedding
    p.add_argument("--embed_dim", type=int, default=4)
    # Checkpointing
    p.add_argument("--model_save_dir", default="saved_models/nba_pt")
    p.add_argument("--model_save_epoch", type=int, default=5)
    p.add_argument("--epoch_continue", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    # Logging
    p.add_argument("--wandb_project", type=str, default="groupnet_nba")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def set_seed(seed, gpu):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(gpu)


LOSS_KEYS = ("total", "pred", "recover", "kl", "diverse")


def _step(model, data, optimizer, split):
    """Single batch forward (+ backward if training). Returns 5-tuple of floats."""
    if split == "train":
        total_loss, l_pred, l_rec, l_kl, l_div = model(data)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    else:
        with torch.no_grad():
            total_loss, l_pred, l_rec, l_kl, l_div = model(data)
    return total_loss.item(), l_pred, l_rec, l_kl, l_div


def run_one_epoch(model, loader, optimizer, args, epoch, split):
    """Run one full pass over `loader`. Returns dict of avg losses."""
    model.train() if split == "train" else model.eval()

    sums = dict.fromkeys(LOSS_KEYS, 0.0)
    n_batches = 0
    total_iter = len(loader)

    for it, data in enumerate(loader):
        l_total, l_pred, l_rec, l_kl, l_div = _step(model, data, optimizer, split)

        sums["total"] += l_total
        sums["pred"] += l_pred
        sums["recover"] += l_rec
        sums["kl"] += l_kl
        sums["diverse"] += l_div
        n_batches += 1

        if split == "train" and it % args.iternum_print == 0:
            print(
                f"Epoch {epoch:03d}/{args.num_epochs:03d} | "
                f"It {it:04d}/{total_iter:04d} | "
                f"total {l_total:.4f} | pred {l_pred:.4f} | "
                f"rec {l_rec:.4f} | kl {l_kl:.4f} | div {l_div:.4f}"
            )

    n = max(n_batches, 1)
    return {k: v / n for k, v in sums.items()}


def main():
    args = parse_args()
    set_seed(args.seed, args.gpu)

    device = (
        torch.device("cuda", index=args.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device:", device)
    print("args:", vars(args))

    os.makedirs(args.model_save_dir, exist_ok=True)

    # Data
    train_files, val_files = load_split_files(args.split_path)
    print(f"split: {len(train_files)} train files, {len(val_files)} val files")

    mu, sigma = compute_xy_stats(train_files)
    print(f"norm stats: mu={mu.tolist()}, sigma={sigma.tolist()}")
    torch.save(
        {"mu": mu, "sigma": sigma},
        os.path.join(args.model_save_dir, "norm_stats.pt"),
    )

    train_set = GroupNetNBAPTDataset(
        train_files,
        mu,
        sigma,
        args.past_length,
        args.future_length,
    )
    val_set = GroupNetNBAPTDataset(
        val_files,
        mu,
        sigma,
        args.past_length,
        args.future_length,
    )
    print(f"usable sequences: train={len(train_set)}, val={len(val_set)}")

    train_sampler = WindowSampler(
        args.batch_size, train_set.max_start, seed=args.seed, shuffle=True
    )
    val_sampler = WindowSampler(
        args.batch_size, val_set.max_start, seed=args.seed, shuffle=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=groupnet_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=groupnet_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = GroupNetWithID(args, device, embed_dim=args.embed_dim)
    model.set_device(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.decay_gamma
    )

    if args.epoch_continue > 0:
        ckpt_path = os.path.join(args.model_save_dir, f"{args.epoch_continue}.p")
        print("resuming from:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    # Train
    # Mirrors network_ml_project's wandb pattern (ref_script.py: wandb.init + wandb.log).
    # Set WANDB_MODE=disabled (or =offline) at runtime to silence wandb without code changes.
    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    ):
        for epoch in range(args.epoch_continue, args.num_epochs):
            train_sampler.set_epoch(epoch)
            train_avg = run_one_epoch(
                model,
                train_loader,
                optimizer,
                args,
                epoch,
                split="train",
            )
            scheduler.step()
            model.step_annealer()

            val_avg = run_one_epoch(
                model,
                val_loader,
                optimizer=None,
                args=args,
                epoch=epoch,
                split="val",
            )

            log_payload = {"epoch": epoch}
            log_payload.update({f"train/{k}": v for k, v in train_avg.items()})
            log_payload.update({f"val/{k}": v for k, v in val_avg.items()})
            wandb.log(log_payload)

            print(
                f"Epoch {epoch:03d} done | "
                f"train_total {train_avg['total']:.4f} | val_total {val_avg['total']:.4f} | "
                f"train_pred {train_avg['pred']:.4f} | val_pred {val_avg['pred']:.4f}"
            )

            if (epoch + 1) % args.model_save_epoch == 0:
                ckpt = {
                    "model_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "model_cfg": args,
                    "mu": mu,
                    "sigma": sigma,
                }
                save_path = os.path.join(args.model_save_dir, f"{epoch + 1}.p")
                torch.save(ckpt, save_path)
                print(f"saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
