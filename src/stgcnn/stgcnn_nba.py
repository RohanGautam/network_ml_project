"""
Social-STGCNN for NBA trajectory prediction — consolidated Lightning + W&B pipeline.

Mirrors src/equivariance/eqmotion_nba.py: a single file holding the dataset,
sampler, Lightning module, datamodule and Kaggle submission helper, wired to
Weights & Biases. The Social-STGCNN architecture (Social_STGCNN) and the
bivariate-Gaussian NLL loss are imported from their existing modules.

The model is *stochastic*: it predicts a per-step bivariate Gaussian over each
agent's position rather than a point. We therefore log both the deterministic
(mean-prediction) metrics — ADE/FDE/MSE, comparable to the EqMotion run — and the
stochastic best-of-K metrics (minADE/minFDE/minMSE) which reward the model for
placing probability mass near the ground truth.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import (
    compute_ade,
    compute_fde,
    compute_mse,
    compute_min_ade,
    compute_min_fde,
    compute_min_mse,
)
from stgcnn.model import Social_STGCNN
from stgcnn.loss import bivariate_loss

dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = DATA_DIR / "test" / "test"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)


# ── Data ────────────────────────────────────────────────────────────────────


class NBADataset(Dataset):
    """
    Holds raw per-sequence coordinates and produces windows on demand.

    Unlike the original stgcnn/dataset.py — which materialized every sliding
    window up front — windows are cut lazily so the sampler can draw a fresh
    random start per sequence each epoch (matching the EqMotion pipeline). The
    interaction adjacency A is built from *raw* (un-normalized) distances, as in
    the original Social-STGCNN, then positions are normalized for the network.

    __getitem__ returns:
      X [2, obs_len, N]        — normalized coords, channel-first for the GCN
      Y [pred_len, N, 2]       — normalized target coords
      A [obs_len, N, N]        — exp(-pairwise distance) interaction graph
    """

    def __init__(self, files, context_size, horizon_size, mu, sigma):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.mu = mu.float()
        self.sigma = sigma.float()
        self.load_data(files)

    def load_data(self, files):
        self.sequences = []  # raw coords [T, N, 2]
        self.max_start = []
        for f in files:
            seq = torch.load(f, weights_only=False).float()
            self.sequences.append(seq[:, :, :2])
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index):
        seq_idx, start = index
        coords = self.sequences[seq_idx][start : start + self.window_size]  # [W, N, 2]
        obs_raw = coords[: self.context_size]  # [obs, N, 2]

        # Interaction graph from raw distances (per observed frame).
        dist = torch.cdist(obs_raw, obs_raw)  # [obs, N, N]
        A = torch.exp(-dist)

        coords_norm = (coords - self.mu) / self.sigma
        X = coords_norm[: self.context_size].permute(2, 0, 1)  # [2, obs, N]
        Y = coords_norm[self.context_size :]  # [pred, N, 2]
        return X, Y, A

    def __len__(self):
        return len(self.sequences)


class NBASampler(Sampler):
    """One random window per sequence per epoch (shared with the EqMotion run)."""

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


# ── Gaussian helpers ──────────────────────────────────────────────────────────


def gaussian_mean(pred) -> Tensor:
    """Point prediction (distribution mean) → [pred, B*N, 2], reshaped for metrics."""
    mu_x, mu_y, _, _, _ = pred
    mu = torch.stack([mu_x, mu_y], dim=-1)  # [B, T, N, 2]
    B, T, N, _ = mu.shape
    return mu.permute(1, 0, 2, 3).reshape(T, B * N, 2)


def gaussian_sample(pred, k: int) -> Tensor:
    """
    Draw K samples from the per-step bivariate Gaussian.

    Returns [K, T, B*N, 2]. Uses the Cholesky form of the 2x2 covariance built
    from (sig_x, sig_y, rho) so samples respect the predicted correlation.
    """
    mu_x, mu_y, sig_x, sig_y, rho = pred
    B, T, N = mu_x.shape
    mean = torch.stack([mu_x, mu_y], dim=-1)  # [B, T, N, 2]
    # Cholesky of [[sx^2, rho sx sy], [rho sx sy, sy^2]].
    one_minus = torch.clamp(1 - rho**2, min=1e-6)
    l11 = sig_x
    l21 = rho * sig_y
    l22 = sig_y * torch.sqrt(one_minus)
    eps = torch.randn(k, B, T, N, 2, device=mu_x.device)
    s_x = l11 * eps[..., 0]
    s_y = l21 * eps[..., 0] + l22 * eps[..., 1]
    samples = mean.unsqueeze(0) + torch.stack([s_x, s_y], dim=-1)  # [K, B, T, N, 2]
    return samples.permute(0, 2, 1, 3, 4).reshape(k, T, B * N, 2)


# ── Lightning module ───────────────────────────────────────────────────────────


class NBASTGCNNLightningModel(L.LightningModule):
    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_step: int = 10,
        lr_gamma: float = 0.5,
        n_samples: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = Social_STGCNN(
            in_channels=2,
            hidden_dim=hidden_dim,
            obs_len=context_size,
            pred_len=horizon_size,
        )

    def on_fit_start(self):
        dm = self.trainer.datamodule
        self.register_buffer("mu", dm.mu.to(self.device))
        self.register_buffer("sigma", dm.sigma.to(self.device))

    def forward(self, X: Tensor, A: Tensor):
        return self.net(X, A)

    def training_step(self, batch, batch_idx):
        X, Y, A = batch
        pred = self(X, A)
        loss = bivariate_loss(pred, Y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, A = batch
        pred = self(X, A)
        loss = bivariate_loss(pred, Y)

        B, T, N, _ = Y.shape
        target = Y.permute(1, 0, 2, 3).reshape(T, B * N, 2)
        target_real = target * self.sigma + self.mu

        # Deterministic (mean-prediction) metrics — comparable to EqMotion.
        mean_pred = gaussian_mean(pred) * self.sigma + self.mu
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            "val/ade_ft",
            compute_ade(mean_pred, target_real),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/fde_ft",
            compute_fde(mean_pred, target_real),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/mse_ft",
            compute_mse(mean_pred, target_real),
            on_epoch=True,
            prog_bar=True,
        )

        # Stochastic best-of-K metrics — specific to this probabilistic model.
        k = self.hparams.n_samples
        samples = gaussian_sample(pred, k) * self.sigma + self.mu  # [K, T, B*N, 2]
        self.log(
            f"val/min_ade_ft_k{k}", compute_min_ade(samples, target_real), on_epoch=True
        )
        self.log(
            f"val/min_fde_ft_k{k}", compute_min_fde(samples, target_real), on_epoch=True
        )
        self.log(
            f"val/min_mse_ft_k{k}", compute_min_mse(samples, target_real), on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.lr_step, gamma=self.hparams.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    @torch.no_grad()
    def predict_mean(self, X: Tensor, A: Tensor) -> Tensor:
        """Mean trajectory for a single normalized sample → [pred, N, 2] (normalized)."""
        self.eval()
        mu_x, mu_y, _, _, _ = self(X.to(self.device), A.to(self.device))
        return torch.stack([mu_x, mu_y], dim=-1).squeeze(0).cpu()


# ── DataModule ─────────────────────────────────────────────────────────────────


class NBADataModule(L.LightningDataModule):
    def __init__(
        self,
        split_path,
        batch_size=128,
        context_size=8,
        horizon_size=12,
        seed=0,
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

    def get_kaggle_submission(
        self, model: NBASTGCNNLightningModel, test_dir: str, target_dir: str
    ):
        mu, sigma = self.mu, self.sigma
        all_traj = []
        for f in sorted(os.listdir(test_dir)):
            if not f.endswith(".pt"):
                continue
            seq = torch.load(os.path.join(test_dir, f), weights_only=False).float()
            coords = seq[:, :, :2]  # [obs, N, 2]
            dist = torch.cdist(coords, coords)
            A = torch.exp(-dist).unsqueeze(0)  # [1, obs, N, N]
            X = ((coords - mu) / sigma).permute(2, 0, 1).unsqueeze(0)  # [1, 2, obs, N]

            pred = model.predict_mean(X, A)  # [pred, N, 2] (normalized)
            pred = pred * sigma + mu  # denormalize → feet
            traj = pred[:, :11, :2].reshape(-1)
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
        out = os.path.join(target_dir, f"submission_stgcnn_{timestamp}.csv")
        df.to_csv(out)
        print(f"Submission saved to {out}")
        return out


if __name__ == "__main__":
    L.seed_everything(0)

    data_module = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=128,
    )

    model = NBASTGCNNLightningModel(lr=1e-3)

    wandb_logger = WandbLogger(project="NML_base", name="stgcnn")

    checkpoint = ModelCheckpoint(monitor="val/mse_ft", mode="min", save_top_k=1)
    early_stop = EarlyStopping(monitor="val/mse_ft", patience=20, mode="min")

    trainer = L.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=[checkpoint],
    )

    trainer.fit(model, data_module)

    data_module.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR))
