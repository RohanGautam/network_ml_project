"""
Model 4: EqMotion
Sequence-to-sequence equivariant trajectory prediction.
Takes the full context window at once; no autoregressive rollout.

Key difference from Models 1-3: EqMotion processes the whole context window
simultaneously using DCT-transformed coordinates as geometric features, with
velocity angles computed internally as invariant pattern features.
"""

import sys
from datetime import datetime
from pathlib import Path
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils.metrics import compute_ade, compute_fde, compute_mse
from equivariance.eqmotion import EqMotion

dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)
COURT_IMAGE = PROJECT_ROOT / "src" / "img" / "basketball_court.png"


# ── Data pipeline (identical to other scripts) ────────────────────────────────


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
        y = self.sequences[seq_idx][start + self.context_size : start + self.window_size]
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
        perm = torch.randperm(n, generator=self.generator).tolist() if self.shuffle else list(range(n))
        perm_start = [(i, self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            for idx, max_start in perm_start[k : k + self.batch_size]:
                start = torch.randint(0, max_start + 1, size=(), generator=self.generator)
                yield idx, start

    def __len__(self):
        return len(self.max_start)


class MultiStepMSE:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def compute(self, pred, target) -> Tensor:
        B, T, N, _ = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        loss = sum(self.loss_fn(pred[t], target[t]) for t in range(T))
        return loss / T


# ── EqMotion wrapper ──────────────────────────────────────────────────────────


class NBAEqMotionModel(nn.Module):
    """
    Wraps EqMotion for NBA trajectory prediction.

    Input:  X [B, T_p, N, 6]  — normalized, features [x, y, dx, dy, isplayer, team]
    Output: [T_f, B*N, 2]     — normalized predicted positions

    Shape mapping into EqMotion:
      pos  [B, N, T_p, 2]  — trajectory coordinates (geometric feature)
      vel  [B, N, T_p, 2]  — position differences
      h    [B, N, T_p]     — velocity magnitudes (invariant pattern feature seed)

    EqMotion handles DCT of pos/vel, velocity angle computation from vel, and
    the equivariant coordinate + invariant pattern update loop internally.
    in_node_nf must equal T_p because embedding2 maps vel_angle [B,N,T_p] → hidden.
    """

    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        hidden_nf: int = 64,
        hid_channel: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.model = EqMotion(
            in_node_nf=context_size,   # vel magnitudes: one scalar per context step
            in_edge_nf=0,
            hidden_nf=hidden_nf,
            in_channel=context_size,   # T_p — DCT input length
            hid_channel=hid_channel,   # DCT latent temporal dim
            out_channel=horizon_size,  # T_f — DCT output length
            device="cpu",              # Lightning handles device placement
            act_fn=nn.SiLU(),
            n_layers=n_layers,
            recurrent=True,
        )

    def forward(self, X: Tensor) -> Tensor:
        B, T, N, _ = X.shape
        pos = X[:, :, :, :2].permute(0, 2, 1, 3)   # [B, N, T_p, 2]
        vel = X[:, :, :, 2:4].permute(0, 2, 1, 3)  # [B, N, T_p, 2]
        h = torch.norm(vel, dim=-1)                  # [B, N, T_p]  velocity magnitudes
        x_pred, _ = self.model(h, pos, vel)          # [B, N, T_f, 2]
        # [B, N, T_f, 2] → [T_f, B, N, 2] → [T_f, B*N, 2]
        return x_pred.permute(2, 0, 1, 3).reshape(self.horizon_size, B * N, 2)


# ── Lightning wrapper ─────────────────────────────────────────────────────────


class NBAEqMotionLightningModel(L.LightningModule):
    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        hidden_nf: int = 64,
        hid_channel: int = 16,
        n_layers: int = 4,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = NBAEqMotionModel(context_size, horizon_size, hidden_nf, hid_channel, n_layers)
        self.loss_fn = MultiStepMSE()

    def on_fit_start(self):
        dm = self.trainer.datamodule
        self.register_buffer("mu", dm.mu.to(self.device))
        self.register_buffer("sigma", dm.sigma.to(self.device))

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn.compute(pred, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn.compute(pred, y)
        B, T, N, _ = y.shape
        target_xy = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        pred_real = pred * self.sigma + self.mu
        target_real = target_xy * self.sigma + self.mu
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/ade_ft", compute_ade(pred_real, target_real), on_epoch=True, prog_bar=True)
        self.log("val/fde_ft", compute_fde(pred_real, target_real), on_epoch=True, prog_bar=True)
        self.log("val/mse_ft", compute_mse(pred_real, target_real), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}

    def get_trajectory(self, X: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            pred = self(X.unsqueeze(0).to(self.device)).cpu()  # [T_f, N, 2]
        X[:, :, :2] = X[:, :, :2] * sigma + mu
        pred = pred * sigma + mu
        static = X[-1, :, 4:].unsqueeze(0).repeat(pred.size(0), 1, 1)  # [T_f, N, 2]
        pred = torch.cat([pred, static], dim=-1)                         # [T_f, N, 4]
        X_display = torch.cat([X[:, :, :2], X[:, :, 4:]], dim=-1)       # [T_p, N, 4]
        return torch.cat([X_display, pred], dim=0).detach()              # [T_p+T_f, N, 4]


class NBADataModule(L.LightningDataModule):
    def __init__(self, split_path, batch_size=64, context_size=8, horizon_size=12, seed=0):
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self):
        sampler = NBASampler(
            self.batch_size, self.val_dataset.max_start, seed=self.seed, shuffle=False
        )
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler)

    def get_kaggle_submission(
        self, model: NBAEqMotionLightningModel, test_dir: str, target_dir: str
    ):
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


if __name__ == "__main__":
    L.seed_everything(0)

    data_module = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=64,
    )

    model = NBAEqMotionLightningModel(lr=1e-3)

    wandb_logger = WandbLogger(project="NML_base", name="eqmotion")

    early_stop = EarlyStopping(monitor="val/loss", patience=20, mode="min")

    trainer = L.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=[early_stop],
    )

    trainer.fit(model, data_module)

    data_module.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR))
