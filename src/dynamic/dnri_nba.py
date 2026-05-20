"""
dNRI (Graber & Schwing, CVPR 2020) adapted to the NBA trajectory task.

Reuses NBADataset / NBASampler / NBADataModule from src/equivariance/eqmotion_nba.py
and wraps the upstream DNRI implementation in src/dynamic/dnri_ref/dnri.py.

Training: teacher-forced next-step prediction over the full C+H window via
DNRI.calculate_loss (NLL + KL between learned posterior and learned prior).
Validation/Test: predict_future with C burn-in steps → H future steps,
metrics computed on positions only in real-world feet.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils.metrics import compute_ade, compute_fde, compute_mse

# from equivariance.eqmotion_nba import NBADataModule, NBADataset
from dynamic.dnri_ref.dnri import DNRI

dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = DATA_DIR / "test" / "test"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)


class NBADataset(Dataset):
    def __init__(self, files, context_size, horizon_size, mu, sigma):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.load_data(files, mu, sigma)

    def load_data(self, files, mu, sigma):
        self.sequences = []
        self.max_start = []
        for f in files:
            seq = torch.load(f, weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - mu) / sigma
            vel = torch.zeros_like(seq[:, :, :2])
            vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
            # feature layout: [x, y, dx, dy, isplayer, team]
            seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)
            self.sequences.append(seq)
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index):
        seq_idx, start = index
        X = self.sequences[seq_idx][start : start + self.context_size]
        y = self.sequences[seq_idx][
            start + self.context_size : start + self.window_size
        ]
        return X, y

    def __len__(self):
        return len(self.sequences)


class NBASampler(Sampler):
    def __init__(self, batch_size, max_start, seed=0, shuffle=True):
        self.batch_size = batch_size
        self.max_start = max_start
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.shuffle = shuffle

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n = len(self)
        perm = (
            torch.randperm(n, generator=self.generator).tolist()
            if self.shuffle
            else list(range(n))
        )
        perm_start = [(i, self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            for idx, max_start in perm_start[k : k + self.batch_size]:
                start = torch.randint(
                    0, max_start + 1, size=(), generator=self.generator
                )
                yield idx, start

    def __len__(self):
        return len(self.max_start)


class NBADataModule(L.LightningDataModule):
    def __init__(
        self, split_path, batch_size=64, context_size=8, horizon_size=12, seed=0
    ):
        super().__init__()
        self.split_path = split_path
        self.batch_size = batch_size
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.seed = seed
        self.mu = None
        self.sigma = None

    def setup(self, stage=None):
        manifest = json.loads(Path(self.split_path).read_text())
        data_dir = PROJECT_ROOT / manifest["data_dir"]
        train_files = [data_dir / f for f in manifest["train"]]
        val_files = [data_dir / f for f in manifest["val"]]
        self.mu, self.sigma = self._compute_normalization_statistics(train_files)
        self.train_dataset = NBADataset(
            train_files, self.context_size, self.horizon_size, self.mu, self.sigma
        )
        self.val_dataset = NBADataset(
            val_files, self.context_size, self.horizon_size, self.mu, self.sigma
        )

    def _compute_normalization_statistics(self, files):
        all_pos = []
        for f in files:
            seq = torch.load(f, weights_only=False)
            all_pos.append(seq[:, :, [0, 1]])
        all_pos = torch.cat(all_pos, dim=0)
        return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))

    def train_dataloader(self):
        sampler = NBASampler(
            self.batch_size, self.train_dataset.max_start, seed=self.seed, shuffle=True
        )
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=sampler
        )

    def val_dataloader(self):
        sampler = NBASampler(
            self.batch_size, self.val_dataset.max_start, seed=self.seed, shuffle=False
        )
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler)

    def get_kaggle_submission(self, model, test_dir: str, target_dir: str):
        all_traj = []
        for f in sorted(os.listdir(test_dir)):
            if not f.endswith(".pt"):
                continue
            seq = torch.load(os.path.join(test_dir, f), weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - self.mu) / self.sigma
            vel = torch.zeros_like(seq[:, :, :2])
            vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
            seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)
            traj = model.get_trajectory(seq, self.mu, self.sigma)
            traj = traj[8:, :, :2].reshape(-1)
            all_traj.append([int(f.removesuffix(".pt"))] + traj.tolist())
        df = (
            pd.DataFrame(
                all_traj,
                columns=["id"]
                + [
                    f"entity_{i}_time_{t}_{axis}"
                    for t in range(12)
                    for i in range(11)
                    for axis in ["x", "y"]
                ],
            )
            .set_index("id")
            .sort_index()
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(target_dir, f"solution_{timestamp}.csv"))


def default_dnri_params(num_vars: int, num_edge_types: int, input_size: int) -> dict:
    """Defaults aligned with the upstream basketball experiment."""
    return {
        "num_vars": num_vars,
        "num_edge_types": num_edge_types,
        "input_size": input_size,
        # Sampling
        "gumbel_temp": 0.5,
        "train_hard_sample": False,
        # Teacher forcing (-1 = always force, the original NRI/dNRI training scheme)
        "teacher_forcing_steps": -1,
        "val_teacher_forcing_steps": -1,
        # Loss normalization — divide NLL/KL by (B,) so optimization is scale-invariant
        "normalize_kl": True,
        "normalize_nll": True,
        "normalize_kl_per_var": False,
        "normalize_nll_per_var": False,
        "kl_coef": 1.0,
        "nll_loss_type": "gaussian",
        "prior_variance": 5e-5,
        "add_uniform_prior": False,
        # Encoder
        "encoder_dropout": 0.0,
        "encoder_hidden": 256,
        "encoder_rnn_hidden": 64,
        "encoder_rnn_type": "lstm",
        "encoder_mlp_num_layers": 3,
        "encoder_mlp_hidden": 128,
        "prior_num_layers": 3,
        "prior_hidden_size": 128,
        # Decoder — recurrent NRI decoder; skip first edge type = "no interaction"
        "decoder_hidden": 64,
        "decoder_dropout": 0.0,
        "skip_first": True,
        # Misc
        "separate_prior_encoder": False,
        "encoder_save_eval_memory": False,
        # gpu flag is only consulted when add_uniform_prior=True (for log_prior placement)
        "gpu": False,
    }


# ── Model ─────────────────────────────────────────────────────────────────────


class NBADNRIModel(nn.Module):
    """
    dNRI wrapper.
      Train: inputs [B, C+H, N, 4] (pos + vel)  → loss via teacher-forced next-step
      Pred:  context [B, C, N, 4]               → futures [B, H, N, 4]
    """

    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        num_vars: int = 11,
        num_edge_types: int = 2,
        input_size: int = 4,
        **overrides,
    ):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        params = default_dnri_params(num_vars, num_edge_types, input_size)
        params.update(overrides)
        self.dnri = DNRI(params)

    def training_loss(self, inputs: Tensor):
        return self.dnri.calculate_loss(inputs, is_train=True, teacher_forcing=True)

    def predict(self, context: Tensor) -> Tensor:
        return self.dnri.predict_future(context, prediction_steps=self.horizon_size)


class NBADNRILightningModel(L.LightningModule):
    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        num_vars: int = 11,
        num_edge_types: int = 2,
        input_size: int = 4,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 1.0,
        **model_overrides,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = NBADNRIModel(
            context_size,
            horizon_size,
            num_vars,
            num_edge_types,
            input_size,
            **model_overrides,
        )

    def on_fit_start(self):
        dm = self.trainer.datamodule
        self.register_buffer("mu", dm.mu.to(self.device))
        self.register_buffer("sigma", dm.sigma.to(self.device))

    @staticmethod
    def _strip_static(t: Tensor) -> Tensor:
        # NBADataset emits [x, y, dx, dy, isplayer, team]; dNRI ingests only the first 4.
        return t[..., :4]

    def _full_window(self, X: Tensor, y: Tensor) -> Tensor:
        return torch.cat([self._strip_static(X), self._strip_static(y)], dim=1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        inputs = self._full_window(X, y)  # [B, C+H, N, 4]
        loss, loss_nll, loss_kl = self.net.training_loss(inputs)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/nll", loss_nll.mean(), on_epoch=True)
        self.log("train/kl", loss_kl.mean(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        context = self._strip_static(X)  # [B, C, N, 4]
        pred = self.net.predict(context)  # [B, H, N, 4]
        pred_pos = pred[..., :2]
        target_pos = y[..., :2]
        B, T, N, _ = target_pos.shape
        pred_flat = pred_pos.permute(1, 0, 2, 3).reshape(T, B * N, 2)
        target_flat = target_pos.permute(1, 0, 2, 3).reshape(T, B * N, 2)
        loss = ((pred_flat - target_flat) ** 2).mean()
        pred_real = pred_flat * self.sigma + self.mu
        target_real = target_flat * self.sigma + self.mu
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            "val/ade_ft",
            compute_ade(pred_real, target_real),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/fde_ft",
            compute_fde(pred_real, target_real),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/mse_ft",
            compute_mse(pred_real, target_real),
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }

    @torch.no_grad()
    def get_trajectory(self, X: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """X: [C, N, 6] normalized → combined [C+H, N, 4] denormalized (pos + static)."""
        self.eval()
        X_dev = X.to(self.device)
        context = self._strip_static(X_dev).unsqueeze(0)  # [1, C, N, 4]
        pred = self.net.predict(context).cpu()  # [1, H, N, 4]
        pred_pos = pred[0, :, :, :2] * sigma + mu  # [H, N, 2]
        X_cpu = X.cpu()
        X_pos = X_cpu[:, :, :2] * sigma + mu  # [C, N, 2]
        static = (
            X_cpu[-1, :, 4:].unsqueeze(0).repeat(pred_pos.size(0), 1, 1)
        )  # [H, N, 2]
        pred_full = torch.cat([pred_pos, static], dim=-1)  # [H, N, 4]
        X_display = torch.cat([X_pos, X_cpu[:, :, 4:]], dim=-1)  # [C, N, 4]
        return torch.cat([X_display, pred_full], dim=0).detach()


# ── Smoke test ────────────────────────────────────────────────────────────────


class _SmokeNBADataModule(NBADataModule):
    """NBADataModule that only loads a handful of files (fast smoke setup)."""

    def __init__(self, split_path, max_train=16, max_val=8, **kwargs):
        super().__init__(split_path=split_path, **kwargs)
        self.max_train = max_train
        self.max_val = max_val

    def setup(self, stage=None):
        manifest = json.loads(Path(self.split_path).read_text())
        data_dir = PROJECT_ROOT / manifest["data_dir"]
        train_files = [data_dir / f for f in manifest["train"][: self.max_train]]
        val_files = [data_dir / f for f in manifest["val"][: self.max_val]]
        self.mu, self.sigma = self._compute_normalization_statistics(train_files)
        self.train_dataset = NBADataset(
            train_files, self.context_size, self.horizon_size, self.mu, self.sigma
        )
        self.val_dataset = NBADataset(
            val_files, self.context_size, self.horizon_size, self.mu, self.sigma
        )


def smoke_test():
    """End-to-end sanity: tiny datamodule, fast_dev_run, predict, submission shape."""
    print("=" * 60)
    print("SMOKE TEST: dNRI for NBA")
    print("=" * 60)

    L.seed_everything(0)

    data_module = _SmokeNBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=4,
        max_train=16,
        max_val=8,
    )
    data_module.setup()

    print(f"train sequences: {len(data_module.train_dataset.sequences)}")
    print(f"val sequences:   {len(data_module.val_dataset.sequences)}")
    print(f"mu={data_module.mu.tolist()}, sigma={data_module.sigma.tolist()}")

    model = NBADNRILightningModel(
        context_size=8,
        horizon_size=12,
        encoder_hidden=64,  # shrink model for smoke
        encoder_rnn_hidden=32,
        decoder_hidden=32,
        encoder_mlp_hidden=32,
        prior_hidden_size=32,
        lr=5e-4,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    trainer = L.Trainer(
        fast_dev_run=2,  # 2 train + 2 val batches
        accelerator="auto",
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module)
    print("  ✓ fast_dev_run (2 train + 2 val) passed")

    # Standalone predict on a fresh test tensor
    device = next(model.parameters()).device
    ctx = torch.randn(2, 8, 11, 4, device=device)
    with torch.no_grad():
        pred = model.net.predict(ctx)
    assert pred.shape == (2, 12, 11, 4), f"predict shape {pred.shape}"
    assert torch.isfinite(pred).all(), "NaN/Inf in predictions"
    print(f"  ✓ net.predict shape={tuple(pred.shape)}, finite=True")

    # Try a real test file end-to-end through get_trajectory
    test_files = sorted(os.listdir(TEST_DIR))[:2]
    for f in test_files:
        seq = torch.load(TEST_DIR / f, weights_only=False)  # [8, 11, 4]
        seq[:, :, [0, 1]] = (
            seq[:, :, [0, 1]].clone() - data_module.mu
        ) / data_module.sigma
        vel = torch.zeros_like(seq[:, :, :2])
        vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
        seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)  # [8, 11, 6]
        traj = model.get_trajectory(seq, data_module.mu, data_module.sigma)
        assert traj.shape == (20, 11, 4), f"get_trajectory shape {traj.shape}"
        assert torch.isfinite(traj).all(), "NaN/Inf in trajectory"
    print(f"  ✓ get_trajectory on {len(test_files)} real test files")

    # Mini Kaggle submission to /tmp
    tmp_dir = Path("/tmp/dnri_smoke_sub")
    tmp_dir.mkdir(exist_ok=True)

    # restrict to a few test files for speed
    class _ScopedTestDir:
        def __init__(self, src, n):
            self.src = src
            self.n = n
            self.dst = Path("/tmp/dnri_smoke_test")

        def __enter__(self):
            self.dst.mkdir(exist_ok=True)
            for f in sorted(os.listdir(self.src))[: self.n]:
                src_path = Path(self.src) / f
                dst_path = self.dst / f
                if not dst_path.exists():
                    dst_path.symlink_to(src_path)
            return str(self.dst)

        def __exit__(self, *a):
            pass

    with _ScopedTestDir(str(TEST_DIR), 3) as scoped:
        data_module.get_kaggle_submission(model, scoped, str(tmp_dir))
    submissions = sorted(tmp_dir.glob("solution_*.csv"))
    assert submissions, "no submission written"
    df = pd.read_csv(submissions[-1])
    expected_cols = 1 + 11 * 12 * 2  # id + entity_i_time_t_axis
    assert df.shape[1] == expected_cols, f"cols {df.shape[1]} != {expected_cols}"
    assert df.shape[0] == 3, f"rows {df.shape[0]} != 3"
    assert df.iloc[:, 1:].notna().all().all(), "NaN values in submission"
    print(f"  ✓ Kaggle submission shape={df.shape}, no NaNs ({submissions[-1].name})")

    print()
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


# ── Full training entrypoint ──────────────────────────────────────────────────


def train(args):
    L.seed_everything(args.seed)

    data_module = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = NBADNRILightningModel(
        context_size=8,
        horizon_size=12,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    callbacks = [EarlyStopping(monitor="val/loss", patience=20, mode="min")]
    if args.wandb:
        logger = WandbLogger(project="NML_base", name=args.run_name or "dnri")
    else:
        logger = False

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)

    if args.submit:
        data_module.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="Run smoke tests and exit."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--submit", action="store_true", help="Write Kaggle submission after training."
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        train(args)
