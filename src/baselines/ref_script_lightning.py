from datetime import datetime
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
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
        """Retrieve sequence given an index in format: (seq_idx:int,start_point:int)."""
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
                f"Dimension mismatch between pred and target ({pred.size(1)} vs {target.size(1)}), check the horizon length. "
            )
        loss = 0
        for t in range(T):
            loss += self.loss_fn(pred[t, :, :], target[t, :, :])
        return loss / T


class RNN(torch.nn.Module):
    """GRU unit for sequence prediction."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.has_pre_norm = True
        self.pre_norm = torch.nn.LayerNorm(input_dim)
        self.GRU_Z = torch.nn.Sequential(
            torch.nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            torch.nn.Sigmoid(),
        )
        self.GRU_R = torch.nn.Sequential(
            torch.nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            torch.nn.Sigmoid(),
        )
        self.GRU_H_Tilde = torch.nn.Sequential(
            torch.nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            torch.nn.Tanh(),
        )

    def forward(self, x, H_prev):
        if self.has_pre_norm:
            x = self.pre_norm(x)
        X = x
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_out = Z * H_prev + (1 - Z) * H_tilde
        return H_out


class NBAModel(torch.nn.Module):
    """Module to perform predictions on the NBA dataset."""

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
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )
        self.context_size = context_size
        self.horizon_size = horizon_size

    def forward(self, X: Tensor) -> Tensor:
        B, _, N, F = X.shape
        h_prev = torch.zeros(size=(B * N, self.RNN.state_dim), device=X.device)
        T = self.context_size + self.horizon_size
        all_preds = []
        for t in range(T):
            if t < self.context_size:
                x = X[:, t, :, :].reshape(B * N, F)
            else:
                x = torch.cat([x, X[:, 0, :, 2:].reshape(B * N, 2)], dim=1)
            h = self.RNN.forward(x, h_prev)
            if t >= self.context_size - 1 and t < T - 1:
                x = self.proj.forward(h)
                all_preds.append(x)
            h_prev = h
        all_preds = torch.stack(all_preds, dim=0)  # [T,B*N,2]
        return all_preds


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
        self.net = NBAModel(
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
        # Reshape target to [T, B*N, 2] to match pred
        B, T, N, _ = y.shape
        target_xy = y[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        # Denormalize to original coordinate space (feet) before computing ADE/FDE
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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

    model = NBALightningModel()

    wandb_logger = WandbLogger(project="NML_base")

    early_stop = EarlyStopping(monitor="val/loss", patience=15, mode="min")

    trainer = L.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator="auto",
        callbacks=[early_stop],
    )

    trainer.fit(model, data_module)

    data_module.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR))
