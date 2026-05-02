"""
Model 3: EGNN-style Geometry-Aware GRU+GNN
GRU temporal backbone + E(n) equivariant graph layer.

Key upgrade over Model 2: messages include pairwise distance d_ij = ||pos_i - pos_j||
(rotation+translation invariant), and positions are updated equivariantly via a
weighted sum of displacement vectors. Inspired by EGNN (Satorras et al., ICML 2021)
and EqMotion (Xu et al., CVPR 2023).
"""

from datetime import datetime
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from torch_geometric.nn import MessagePassing
import dotenv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils.metrics import compute_ade, compute_fde

dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)
COURT_IMAGE = PROJECT_ROOT / "src" / "img" / "basketball_court.png"


# ── Data pipeline (identical to permutation_equivariant.py) ──────────────────


class NBADataset(Dataset):
    """Dataset to load NBA highlights tensor data."""

    def __init__(
        self,
        files,
        context_size: int,
        horizon_size: int,
        mu: Tensor,
        sigma: Tensor,
    ):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.load_data(files, mu, sigma)

    def load_data(self, files, mu: Tensor, sigma: Tensor):
        """Load all sequences and normalize positions."""
        self.sequences = []
        self.max_start = []
        for f in files:
            seq = torch.load(f, weights_only=False)  # [T,N,F]
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - mu) / sigma
            self.sequences.append(seq)
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index) -> tuple:
        seq_idx, start = index
        T = self.window_size
        X = self.sequences[seq_idx][start : start + self.context_size]
        y = self.sequences[seq_idx][start + self.context_size : start + T]
        return X, y

    def __len__(self):
        return len(self.sequences)


class NBASampler(Sampler):
    def __init__(self, batch_size: int, max_start: list, seed=0, shuffle=True):
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
        if self.shuffle:
            perm = torch.randperm(n, generator=self.generator).tolist()
        else:
            perm = list(range(n))
        perm_start = [(i, self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            batch = perm_start[k : k + self.batch_size]
            for idx, max_start in batch:
                start = torch.randint(
                    0, max_start + 1, size=(), generator=self.generator
                )
                yield idx, start

    def __len__(self):
        return len(self.max_start)


class MultiStepMSE:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def compute(self, pred, target) -> Tensor:
        """Compute the average MSE over the horizon."""
        B, T, N, F = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        if pred.size(1) != target.size(1):
            raise ValueError(
                f"Dimension mismatch between pred and target ({pred.size(1)} vs {target.size(1)})"
            )
        loss = 0
        for t in range(T):
            loss += self.loss_fn(pred[t, :, :], target[t, :, :])
        return loss / T


# ── GRU cell (identical to other scripts) ────────────────────────────────────


class RNN(nn.Module):
    """GRU unit for sequence prediction."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.has_pre_norm = True
        self.pre_norm = nn.LayerNorm(input_dim)
        self.GRU_Z = nn.Sequential(
            nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            nn.Sigmoid(),
        )
        self.GRU_R = nn.Sequential(
            nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            nn.Sigmoid(),
        )
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            nn.Tanh(),
        )

    def forward(self, x, H_prev):
        if self.has_pre_norm:
            x = self.pre_norm(x)
        Z = self.GRU_Z(torch.cat([x, H_prev], dim=1))
        R = self.GRU_R(torch.cat([x, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([x, R * H_prev], dim=1))
        return Z * H_prev + (1 - Z) * H_tilde


def make_complete_edge_index(N: int, B: int, device) -> Tensor:
    """
    Fully-connected edge index (no self-loops) for a batch of B graphs, each with N nodes.
    Returns shape [2, B * N * (N-1)].
    """
    src = torch.arange(N, device=device).repeat_interleave(N)
    dst = torch.arange(N, device=device).repeat(N)
    mask = src != dst
    ei = torch.stack([src[mask], dst[mask]])  # [2, N*(N-1)]
    E = ei.size(1)
    ei_batched = ei.repeat(1, B)
    offsets = torch.arange(B, device=device).repeat_interleave(E) * N
    return ei_batched + offsets.unsqueeze(0)


# ── EGNN layer ────────────────────────────────────────────────────────────────


class EGNNLayer(MessagePassing):
    """
    E(n)-equivariant graph layer.

    Messages include pairwise distance d_ij (invariant) alongside hidden states.
    Positions are updated via a weighted sum of displacement vectors (equivariant).
    Hidden states are updated via a residual MLP on aggregated messages.

    Returns updated (h, pos) — permutation equivariant and translation equivariant.
    """

    def __init__(self, state_dim: int):
        super().__init__(aggr="add")
        self.state_dim = state_dim
        hidden = state_dim * 2

        self.msg_mlp = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
            nn.SiLU(),
        )

        # Small init on coord output prevents exploding position updates early in training
        coord_out = nn.Linear(state_dim, 1, bias=False)
        nn.init.xavier_uniform_(coord_out.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            coord_out,
            nn.Tanh(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.SiLU(),
        )

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor):
        """
        h:          [B*N, state_dim]
        pos:        [B*N, 2]  — normalized xy positions
        edge_index: [2, E]
        Returns (h_new, pos_new), both [B*N, ...].
        """
        # propagate returns [B*N, state_dim + 2]: aggregated messages || aggregated weighted diffs
        agg = self.propagate(edge_index, h=h, pos=pos)
        agg_m = agg[:, : self.state_dim]     # [B*N, state_dim]
        agg_coord = agg[:, self.state_dim :]  # [B*N, 2]

        h_new = h + self.update_mlp(torch.cat([h, agg_m], dim=-1))
        pos_new = pos + agg_coord
        return h_new, pos_new

    def message(self, h_i: Tensor, h_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        diff = pos_i - pos_j                                        # [E, 2]
        d = torch.norm(diff, dim=-1, keepdim=True)                  # [E, 1]
        m = self.msg_mlp(torch.cat([h_i, h_j, d], dim=-1))         # [E, state_dim]
        w = self.coord_mlp(m)                                       # [E, 1]
        return torch.cat([m, diff * w], dim=-1)                     # [E, state_dim + 2]


# ── EGNN model ────────────────────────────────────────────────────────────────


class NBAEGNNModel(nn.Module):
    """
    GRU + EGNN: geometry-aware multi-agent trajectory prediction.

    At each timestep:
      1. GRU step encodes per-agent temporal dynamics.
      2. EGNNLayer exchanges messages weighted by pairwise distances and
         updates positions equivariantly.

    Position prediction comes directly from the equivariant coord update,
    not a separate linear projection head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        context_size: int,
        horizon_size: int,
    ):
        super().__init__()
        self.RNN = RNN(input_dim, state_dim)
        self.egnn = EGNNLayer(state_dim)
        self.context_size = context_size
        self.horizon_size = horizon_size

    def forward(self, X: Tensor) -> Tensor:
        B, _, N, F = X.shape
        h_prev = torch.zeros(size=(B * N, self.RNN.state_dim), device=X.device)
        edge_index = make_complete_edge_index(N, B, X.device)
        T = self.context_size + self.horizon_size
        all_preds = []

        pos = X[:, 0, :, :2].reshape(B * N, 2)  # initialise from first frame

        for t in range(T):
            if t < self.context_size:
                x = X[:, t, :, :].reshape(B * N, F)
                pos = X[:, t, :, :2].reshape(B * N, 2)  # ground-truth pos during context
            else:
                # During prediction: feed [predicted_pos, static_features] as input
                x = torch.cat([pos, X[:, 0, :, 2:].reshape(B * N, 2)], dim=1)

            h = self.RNN.forward(x, h_prev)           # [B*N, D]
            h, pos = self.egnn(h, pos, edge_index)    # equivariant update

            if t >= self.context_size - 1 and t < T - 1:
                all_preds.append(pos)  # equivariant coord IS the position prediction

            h_prev = h

        return torch.stack(all_preds, dim=0)  # [H, B*N, 2]


# ── Lightning wrappers ────────────────────────────────────────────────────────


class NBALightningModel(L.LightningModule):
    ENTITY_MAPPING = {-1: "Team_A", 0: "Ball", 1: "Team_B"}

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 2,
        state_dim: int = 32,
        context_size: int = 8,
        horizon_size: int = 12,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = NBAEGNNModel(
            input_dim, output_dim, state_dim, context_size, horizon_size
        )
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
        ade = compute_ade(pred_real, target_real)
        fde = compute_fde(pred_real, target_real)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/ade_ft", ade, on_epoch=True, prog_bar=True)
        self.log("val/fde_ft", fde, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }

    def get_trajectory(self, X: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """Generate the predicted trajectory for a single sequence."""
        self.eval()
        with torch.no_grad():
            pred = self(X.unsqueeze(0).to(self.device)).cpu()
        X[:, :, :2] = X[:, :, :2] * sigma + mu
        pred = pred * sigma + mu
        static = X[-1, :, 2:].unsqueeze(0).repeat(pred.size(0), 1, 1)
        pred = torch.cat([pred, static], dim=-1)
        return torch.cat([X, pred], dim=0).detach()

    def animate_sequence(
        self, sequence: Tensor, interval: int = 50, pred_seq: Tensor = None
    ):
        """Animate the given sequence."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 95)
        ax.set_ylim(0, 50)
        court_image = plt.imread(str(COURT_IMAGE))
        ax.imshow(
            court_image, extent=ax.get_xlim() + ax.get_ylim(), aspect="auto", zorder=-1
        )
        entities = sequence[:, :, -1].unique()
        color_palette = dict(
            zip(
                ["GT_" + self.ENTITY_MAPPING[e.item()] for e in entities],
                ["#BBDEFB", "#000000", "#1E88E5"],
            )
        )
        if pred_seq is not None:
            pred_palette = dict(
                zip(
                    ["Pred_" + self.ENTITY_MAPPING[e.item()] for e in entities],
                    ["#FFCDD2", "#FFFFFF", "#D32F2F"],
                )
            )
            color_palette = color_palette | pred_palette
        scatters = {
            entity: ax.scatter([], [], s=40, color=color, label=entity)
            for entity, color in color_palette.items()
        }
        ax.legend()

        def update_scatter(frame, scatters):
            for entity_name, scatter in scatters.items():
                entity_id = [
                    e_id
                    for e_id, e_name in self.ENTITY_MAPPING.items()
                    if e_name in entity_name
                ][0]
                frame_data = (
                    sequence[frame].numpy()
                    if "GT" in entity_name
                    else pred_seq[frame].numpy()
                )
                if "GT" in entity_name or frame > 8:
                    entity_mask = frame_data[:, -1] == entity_id
                    entity_data = frame_data[entity_mask].copy()
                    entity_data[:, 0] += 47.5
                    entity_data[:, 1] += 25
                    scatter.set_offsets(entity_data[:, [0, 1]])
            return scatters.values()

        def update(frame):
            fig.suptitle(f"Frame {frame + 1}")
            return update_scatter(frame, scatters)

        ani = FuncAnimation(
            fig, update, frames=len(sequence), interval=interval, blit=True
        )
        plt.close(fig)
        return HTML(ani.to_jshtml())


class NBADataModule(L.LightningDataModule):
    def __init__(
        self,
        split_path: str,
        batch_size: int = 64,
        context_size: int = 8,
        horizon_size: int = 12,
        seed: int = 0,
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

    def _compute_normalization_statistics(self, files) -> tuple:
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
        self, model: NBALightningModel, test_dir: str, target_dir: str
    ):
        """Generate Kaggle submission CSV from the trained model."""
        all_traj = []
        for f in sorted(os.listdir(test_dir)):
            if not f.endswith(".pt"):
                continue
            seq = torch.load(os.path.join(test_dir, f), weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - self.mu) / self.sigma
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

    model = NBALightningModel(lr=3e-4)

    wandb_logger = WandbLogger(project="NML_base", name="egnn_stable")

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
