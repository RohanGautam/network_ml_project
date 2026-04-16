# %% [markdown]
# # Data Processing

# %%
from datetime import datetime
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import wandb
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)
COURT_IMAGE = PROJECT_ROOT / "src" / "img" / "basketball_court.png"

# %% [markdown]
# ### Public

# %%
# Data classes


class NBADataset(Dataset):
    """Dataset to load NBA highlights tensor data."""

    def __init__(
        self,
        data_path: str,
        context_size: int,
        horizon_size: int,
        mu: Tensor,
        sigma: Tensor,
    ):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.load_data(data_path, mu, sigma)

    def load_data(self, data_path: str, mu: Tensor, sigma: Tensor):
        """Load all sequences and normalize positions."""
        self.sequences = []
        self.max_start = []
        for f in [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".pt")
        ]:
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
        # Data attributes
        self.batch_size = batch_size
        self.max_start = max_start
        # Random generator
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.shuffle = shuffle

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        # Randomize order of samples
        n = len(self)
        if self.shuffle:
            perm = torch.randperm(n, generator=self.generator).tolist()
        else:
            perm = list(range(n))
        perm_start = [(i, self.max_start[i]) for i in perm]
        # Do block partition at the batch size
        for k in range(0, n, self.batch_size):
            batch = perm_start[k : k + self.batch_size]
            for idx, max_start in batch:
                start = torch.randint(
                    0, max_start + 1, size=(), generator=self.generator
                )
                yield idx, start

    def __len__(self):
        return len(self.max_start)


# Loss


class MultiStepMSE:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def compute(self, pred, target) -> Tensor:
        """Compute the average MSE over the horizon."""
        # Verify input
        B, T, N, F = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        if pred.size(1) != target.size(1):
            raise ValueError(
                f"Dimension mismatch between pred and target ({pred.size(1)} vs {target.size(1)}), check the horizon length. "
            )
        # Compute loss
        loss = 0
        for t in range(T):
            loss += self.loss_fn(pred[t, :, :], target[t, :, :])
        return loss / T


# Model


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
        # Modules
        self.RNN = RNN(input_dim, state_dim)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )
        # Time attributes
        self.context_size = context_size
        self.horizon_size = horizon_size

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass."""
        B, _, N, F = X.shape
        h_prev = torch.zeros(size=(B * N, self.RNN.state_dim), device=X.device)
        T = self.context_size + self.horizon_size
        all_preds = []
        for t in range(T):
            # Format input
            if t < self.context_size:
                x = X[:, t, :, :].reshape(B * N, F)
            else:
                x = torch.cat([x, X[:, 0, :, 2:].reshape(B * N, 2)], dim=1)
            # Process input
            h = self.RNN.forward(x, h_prev)
            if t >= self.context_size - 1 and t < T - 1:
                x = self.proj.forward(h)
                all_preds.append(x)
            # Update state
            h_prev = h
        all_preds = torch.stack(all_preds, dim=0)  # [T,B*N,2]
        return all_preds


class RNN(torch.nn.Module):
    """GRU unit for sequence prediction."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        # RNN
        self.state_dim = state_dim
        self.has_pre_norm = True
        self.pre_norm = torch.nn.LayerNorm(input_dim)
        # update gate.
        self.GRU_Z = torch.nn.Sequential(
            torch.nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            torch.nn.Sigmoid(),
        )
        # reset gate.
        self.GRU_R = torch.nn.Sequential(
            torch.nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            torch.nn.Sigmoid(),
        )
        # new embedding gate.
        self.GRU_H_Tilde = torch.nn.Sequential(
            torch.nn.Linear(input_dim + self.state_dim, self.state_dim, bias=True),
            torch.nn.Tanh(),
        )

    def forward(self, x, H_prev):
        # Normalize
        if self.has_pre_norm:
            x = self.pre_norm(x)
        # Compute output from GRU module.
        X = x
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_out = Z * H_prev + (1 - Z) * H_tilde
        return H_out


# Training pipeline


class NBATrainer:
    ENTITY_MAPPING = {-1: "Team_A", 0: "Ball", 1: "Team_B"}

    def __init__(
        self,
        batch_size: int,
        epochs: int,
        train_path: str,
        val_path: str,
        device: str = "cuda",
        context_size: int = 8,
        horizon_size: int = 12,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.context_size = context_size
        self.device = device
        self.epochs = epochs
        self.horizon_size = horizon_size
        self.train_path = train_path
        self.val_path = val_path
        self.seed = seed
        # Load data
        mu, sigma = self.compute_normalization_statistics(self.train_path)
        self.mu, self.sigma = mu, sigma
        self.train_dataset = NBADataset(
            self.train_path, context_size, horizon_size, mu, sigma
        )
        self.val_dataset = NBADataset(
            self.val_path, context_size, horizon_size, mu, sigma
        )

    def compute_normalization_statistics(self, data_path: str) -> tuple:
        """Compute mu and sigma for normalization of spatial positions."""
        # Load all positions
        all_pos = []
        for f in [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".pt")
        ]:
            seq = torch.load(f, weights_only=False)  # [T,N,F]
            pos = seq[:, :, [0, 1]]
            all_pos.append(pos)
        # Compute mean and standard deviation
        all_pos = torch.cat(all_pos, dim=0)  # [L,N,2], L = T1+T2+...
        mu = all_pos.mean(dim=(0, 1))
        sigma = all_pos.std(dim=(0, 1))
        return mu, sigma

    def set_seed(self) -> torch.Generator:
        """Setup reproducibility for a set of libraries."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        return generator

    def create_sweep_config(self) -> dict:
        """Create the sweep config for the wandb grid search."""
        sweep_config = {"method": "grid"}
        metric = {"name": "train_loss", "goal": "minimize"}
        sweep_config["metric"] = metric
        sweep_config["parameters"] = {
            "epochs": {"value": self.epochs},
            "batch_size": {"value": self.batch_size},
        }
        return sweep_config

    def pipeline(self):
        """Launch training pipeline with logging."""
        sweep_config = self.create_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project="NML_base")
        wandb.agent(sweep_id, self.train)

    def train(
        self,
    ):
        """Basic training pipeline."""
        # Create dataloader
        generator = self.set_seed()
        train_sampler = NBASampler(
            batch_size=self.batch_size,
            max_start=self.train_dataset.max_start,
            shuffle=True,
        )
        val_sampler = NBASampler(
            batch_size=self.batch_size,
            max_start=self.val_dataset.max_start,
            shuffle=False,
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            generator=generator,
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            generator=generator,
        )
        # Model and optimizer
        model = NBAModel(4, 2, 32, self.context_size, self.horizon_size).to(self.device)
        loss = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
        # Model training
        with wandb.init(
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            )
        ):
            dataloader_dict = {"train": train_dataloader, "val": val_dataloader}
            for epoch in range(self.epochs):
                for split, dataloader in dataloader_dict.items():
                    if split == "val":
                        model.eval()
                        with torch.no_grad():
                            self.train_one_epoch(
                                dataloader, loss, model, optimizer, scheduler, split
                            )
                    else:
                        model.train()
                        self.train_one_epoch(
                            dataloader, loss, model, optimizer, scheduler, split
                        )
                    sampler = dataloader.sampler
                    sampler.set_epoch(epoch)
                    scheduler.step()
            torch.save(model.state_dict(), os.path.join(f"model.pth"))

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: MultiStepMSE,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        split: str,
    ):
        """Single epoch for the model."""
        # Set model mode
        if split == "train":
            model.train()
        else:
            model.eval()
        # Iterate over batches
        avg_loss = 0
        for batch in dataloader:
            # Forward pass
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)
            pred = model(X)
            # Optimize model
            loss = loss_fn.compute(pred, y)
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            avg_loss += loss
        # Log
        avg_loss = avg_loss / len(dataloader)
        wandb.log({f"{split}_loss": avg_loss.item()})

    def get_trajectory(self, X: Tensor) -> Tensor:
        """Generate the predicted trajectory using the trained model."""
        # Forward
        model = NBAModel(4, 2, 32, 8, 12).to(self.device)
        model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=self.device))
        model.eval()
        with torch.no_grad():
            pred = model(X.unsqueeze(0).to(self.device)).cpu()
        # Denormalize
        X[:, :, :2] = X[:, :, :2] * self.sigma + self.mu
        pred = pred * self.sigma + self.mu
        # Format
        static = X[-1, :, 2:].unsqueeze(0).repeat(pred.size(0), 1, 1)
        pred = torch.cat([pred, static], dim=-1)
        pred = torch.cat([X, pred], dim=0).detach()
        return pred

    def get_kaggle_submission(self, test_dir: str, target_dir: str) -> pd.DataFrame:
        """Generate the sequences for a Kaggle submission and save the file in the given directory."""
        # Generate trajectories
        all_traj = []
        for f, f_path in [
            (f, os.path.join(test_dir, f))
            for f in os.listdir(test_dir)
            if f.endswith(".pt")
        ]:
            seq = torch.load(f_path, weights_only=False)  # [T,N,F]
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - self.mu) / self.sigma
            traj = self.get_trajectory(seq)
            traj = traj[8:, :, :2]
            traj = traj.reshape(-1)
            all_traj.append([int(f.removesuffix(".pt"))] + traj.tolist())
        # Format
        all_traj = pd.DataFrame(
            all_traj,
            columns=["id"]
            + [
                f"entity_{i}_time_{t}_{axis}"
                for t in range(12)
                for i in range(11)
                for axis in ["x", "y"]
            ],
        ).set_index("id")
        all_traj = all_traj.sort_index()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_traj.to_csv(os.path.join(target_dir, f"solution_{timestamp}.csv"))

    def animate_sequence(
        self, sequence: Tensor, interval: int = 50, pred_seq: Tensor = None
    ):
        """Animate the given sequence."""
        # Setup background
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 95)
        ax.set_ylim(0, 50)
        court_image = plt.imread(str(COURT_IMAGE))
        ax.imshow(
            court_image, extent=ax.get_xlim() + ax.get_ylim(), aspect="auto", zorder=-1
        )
        # Color mapping for ground truth
        entities = sequence[:, :, -1].unique()
        color_palette = dict(
            zip(
                ["GT_" + self.ENTITY_MAPPING[e.item()] for e in entities],
                ["#BBDEFB", "#000000", "#1E88E5"],
            )
        )
        # Color mapping for predictions if available
        if pred_seq is not None:
            pred_palette = dict(
                zip(
                    ["Pred_" + self.ENTITY_MAPPING[e.item()] for e in entities],
                    ["#FFCDD2", "#FFFFFF", "#D32F2F"],
                )
            )
            color_palette = color_palette | pred_palette
        # Create scatter object
        scatters = dict()
        for entity, color in color_palette.items():
            scatters[entity] = ax.scatter([], [], s=40, color=color, label=entity)
        ax.legend()

        # Animation update
        def update_scatter(frame, scatters):
            for entity_name, scatter in scatters.items():
                # Retrieve sequence values
                entity_id = [
                    e_id
                    for e_id, e_name in self.ENTITY_MAPPING.items()
                    if e_name in entity_name
                ][0]
                if "GT" in entity_name:
                    frame_data = sequence[frame].numpy()
                else:
                    frame_data = pred_seq[frame].numpy()
                # Format for plotting
                if "GT" in entity_name or frame > 8:
                    entity_mask = frame_data[:, -1] == entity_id
                    entity_data = frame_data[entity_mask]
                    entity_data[:, 0] += 47.5
                    entity_data[:, 1] += 25
                    scatter.set_offsets(entity_data[:, [0, 1]])
            return scatters.values()

        def update(frame):
            fig.suptitle(f"Frame {frame + 1}")
            points = update_scatter(frame, scatters)
            return points

        # Create animation
        ani = FuncAnimation(
            fig, update, frames=len(sequence), interval=interval, blit=True
        )
        plt.close(fig)
        return HTML(ani.to_jshtml())


if __name__ == "__main__":
    nba_trainer = NBATrainer(
        batch_size=64,
        epochs=10,
        train_path=str(TRAIN_DIR),
        val_path=str(TRAIN_DIR),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
    )

    # Train
    nba_trainer.train()

    # Generate submission from the trained model
    nba_trainer.get_kaggle_submission(str(TEST_DIR), str(SUBMISSION_DIR))
