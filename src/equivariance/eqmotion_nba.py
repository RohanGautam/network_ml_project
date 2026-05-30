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


# Court-frame landmark presets, each as a list of (x_ft, y_ft, team_id) in the
# raw court-centered frame. Distinct team_ids let the model's id_embed learn a
# separate embedding per landmark type. All sets are chosen to respect the
# court's D2 symmetry (under 180° rotation and the two axis reflections each
# set maps to itself), so they don't break EqMotion's equivariance.
LANDMARK_SETS = {
    "hoops":   [(-41.75, 0.0, 3.0), (41.75, 0.0, 3.0)],
    "ft":      [(-28.0,  0.0, 4.0), (28.0,  0.0, 4.0)],     # free-throw lines
    "3pt":     [(-18.0,  0.0, 5.0), (18.0,  0.0, 5.0)],     # 3-pt arc apex
    "corners": [(-47.0, -25.0, 6.0), (-47.0, 25.0, 6.0),
                ( 47.0, -25.0, 6.0), ( 47.0, 25.0, 6.0)],
    "center":  [(0.0, 0.0, 7.0)],
}


def landmarks_from_spec(spec):
    """Parse 'hoops,ft' → concatenated list of (x,y,team_id). Empty → []."""
    if not spec or spec == "none":
        return []
    out = []
    for name in spec.split(","):
        name = name.strip()
        if name not in LANDMARK_SETS:
            raise ValueError(
                f"unknown landmark preset '{name}'; available: {list(LANDMARK_SETS)}"
            )
        out.extend(LANDMARK_SETS[name])
    return out


class NBADataset(Dataset):
    def __init__(self, files, context_size, horizon_size, mu, sigma,
                 add_hoops=False, landmarks=None):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        # `landmarks` is the general form (list of (x,y,team_id)). `add_hoops` is
        # kept as a back-compat alias for landmarks=LANDMARK_SETS["hoops"].
        if landmarks is None:
            landmarks = LANDMARK_SETS["hoops"] if add_hoops else []
        self.landmarks = landmarks
        self.load_data(files, mu, sigma)

    def load_data(self, files, mu, sigma):
        self.sequences = []
        self.max_start = []
        # Precompute normalized landmark positions once. Each landmark contributes
        # one static node appended to the end of the agent axis with [x,y]=normed
        # position, zero velocity, isplayer=0, team=its preset team_id.
        if self.landmarks:
            raw_land = torch.tensor([[x, y] for x, y, _ in self.landmarks],
                                    dtype=torch.float32)
            norm_land = (raw_land - mu) / sigma
            team_ids = torch.tensor([t for _, _, t in self.landmarks],
                                    dtype=torch.float32)
        for f in files:
            seq = torch.load(f, weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - mu) / sigma
            vel = torch.zeros_like(seq[:, :, :2])
            vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
            # feature layout: [x, y, dx, dy, isplayer, team]
            seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)

            if self.landmarks:
                # Append L static landmark nodes: [T, 11, 6] -> [T, 11+L, 6]
                T = seq.shape[0]
                L = len(self.landmarks)
                land_nodes = torch.zeros((T, L, 6), dtype=seq.dtype)
                land_nodes[:, :, :2] = norm_land
                land_nodes[:, :, 4] = 0.0      # isplayer = 0
                land_nodes[:, :, 5] = team_ids  # one team_id per landmark type
                seq = torch.cat([seq, land_nodes], dim=1)

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


class NBAEvalSampler(Sampler):
    """Deterministic validation windows for a stable, low-variance val metric.

    Instead of one *random* window per sequence per epoch (which makes val/mse_ft
    bounce between epochs and yields a noisy checkpoint-selection signal), this
    enumerates a fixed set of evenly-spaced windows per sequence. Each sequence
    contributes the same number of windows (capped at its valid count), so long
    sequences are not over-weighted — mirroring the test set's one-window-per-id
    structure while averaging out window-position variance.
    """

    def __init__(self, max_start, windows_per_seq=8):
        self.windows = []
        for i, ms in enumerate(max_start):
            if ms <= 0:
                starts = [0]
            else:
                k = min(windows_per_seq, ms + 1)
                starts = sorted(
                    {int(round(s)) for s in torch.linspace(0, ms, k).tolist()}
                )
            self.windows.extend((i, s) for s in starts)

    def __iter__(self):
        return iter(self.windows)

    def __len__(self):
        return len(self.windows)


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
            in_node_nf=context_size,  # vel magnitudes: one scalar per context step
            in_edge_nf=0,
            hidden_nf=hidden_nf,
            in_channel=context_size,  # T_p — DCT input length
            hid_channel=hid_channel,  # DCT latent temporal dim
            out_channel=horizon_size,  # T_f — DCT output length
            device="cpu",  # Lightning handles device placement
            act_fn=nn.SiLU(),
            n_layers=n_layers,
            recurrent=True,
            id_dim=2,  # isplayer + team
        )

    def forward(self, X: Tensor) -> Tensor:
        B, T, N, _ = X.shape
        pos = X[:, :, :, :2].permute(0, 2, 1, 3)  # [B, N, T_p, 2]
        vel = X[:, :, :, 2:4].permute(0, 2, 1, 3)  # [B, N, T_p, 2]
        h = torch.norm(vel, dim=-1)  # [B, N, T_p]  velocity magnitudes
        agent_id = X[:, 0, :, 4:]  # [B, N, 2]  isplayer + team (static)
        x_pred, _ = self.model(h, pos, vel, agent_id=agent_id)  # [B, N, T_f, 2]
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
        weight_decay: float = 5e-4,
        lr_scheduler: str = "none",  # "none" | "cosine" | "plateau"
        max_epochs: int = 500,
        warmup_epochs: int = 0,
        n_landmarks: int = 0,  # static court nodes appended last (e.g. 2 hoops)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = NBAEqMotionModel(
            context_size, horizon_size, hidden_nf, hid_channel, n_layers
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
        # Score only the real entities (players + ball); static landmark nodes
        # (hoops) are appended last and are ~stationary, so including them in the
        # mean deflates val/mse_ft and breaks comparability with the Kaggle metric
        # (which is over the 11 real entities only).
        n_land = self.hparams.n_landmarks
        if n_land > 0:
            n_real = N - n_land
            pred_real = pred_real.view(T, B, N, 2)[:, :, :n_real, :].reshape(
                T, B * n_real, 2
            )
            target_real = target_real.view(T, B, N, 2)[:, :, :n_real, :].reshape(
                T, B * n_real, 2
            )
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
        sched = self.hparams.lr_scheduler
        if sched == "none":
            return {"optimizer": optimizer}
        if sched == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
            }
        if sched == "cosine":
            warmup = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total - warmup), eta_min=self.hparams.lr * 0.02
            )
            if warmup > 0:
                warm = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warm, cosine], milestones=[warmup]
                )
            else:
                scheduler = cosine
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        raise ValueError(f"unknown lr_scheduler {sched}")

    def get_trajectory(self, X: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            pred = self(X.unsqueeze(0).to(self.device)).cpu()  # [T_f, N, 2]
        X[:, :, :2] = X[:, :, :2] * sigma + mu
        pred = pred * sigma + mu
        static = X[-1, :, 4:].unsqueeze(0).repeat(pred.size(0), 1, 1)  # [T_f, N, 2]
        pred = torch.cat([pred, static], dim=-1)  # [T_f, N, 4]
        X_display = torch.cat([X[:, :, :2], X[:, :, 4:]], dim=-1)  # [T_p, N, 4]
        return torch.cat([X_display, pred], dim=0).detach()  # [T_p+T_f, N, 4]


class NBADataModule(L.LightningDataModule):
    def __init__(
        self,
        split_path,
        batch_size=64,
        context_size=8,
        horizon_size=12,
        seed=0,
        add_hoops=False,
        landmarks=None,
        iso_norm=False,
        full_val=False,
        val_windows_per_seq=8,
    ):
        super().__init__()
        self.split_path = split_path
        self.batch_size = batch_size
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.seed = seed
        # `landmarks` is the general form; `add_hoops` is a back-compat alias.
        if landmarks is None:
            landmarks = LANDMARK_SETS["hoops"] if add_hoops else []
        self.landmarks = landmarks
        self.add_hoops = add_hoops  # kept for any external readers; prefer landmarks
        self.iso_norm = iso_norm
        self.full_val = full_val
        self.val_windows_per_seq = val_windows_per_seq
        self.mu = None
        self.sigma = None

    def setup(self, stage=None):
        manifest = json.loads(Path(self.split_path).read_text())
        data_dir = PROJECT_ROOT / manifest["data_dir"]
        train_files = [data_dir / f for f in manifest["train"]]
        val_files = [data_dir / f for f in manifest["val"]]
        self.mu, self.sigma = self._compute_normalization_statistics(train_files)
        self.train_dataset = NBADataset(
            train_files,
            self.context_size,
            self.horizon_size,
            self.mu,
            self.sigma,
            landmarks=self.landmarks,
        )
        self.val_dataset = NBADataset(
            val_files,
            self.context_size,
            self.horizon_size,
            self.mu,
            self.sigma,
            landmarks=self.landmarks,
        )

    def _compute_normalization_statistics(self, files):
        all_pos = []
        for f in files:
            seq = torch.load(f, weights_only=False)
            all_pos.append(seq[:, :, [0, 1]])
        all_pos = torch.cat(all_pos, dim=0)
        mu = all_pos.mean(dim=(0, 1))
        if self.iso_norm:
            # Shared scalar std across x and y so a physical rotation maps to a
            # rotation in the normalized frame — preserving EqMotion's built-in
            # rotation/reflection equivariance (anisotropic per-axis std breaks it).
            s = all_pos.std()
            sigma = torch.stack([s, s])
        else:
            sigma = all_pos.std(dim=(0, 1))
        return mu, sigma

    def train_dataloader(self):
        sampler = NBASampler(
            self.batch_size, self.train_dataset.max_start, seed=self.seed, shuffle=True
        )
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, sampler=sampler
        )

    def val_dataloader(self):
        if self.full_val:
            # Deterministic, evenly-spaced windows → stable val/mse_ft for robust
            # checkpoint selection (no epoch-to-epoch window-draw noise).
            sampler = NBAEvalSampler(
                self.val_dataset.max_start, windows_per_seq=self.val_windows_per_seq
            )
        else:
            sampler = NBASampler(
                self.batch_size,
                self.val_dataset.max_start,
                seed=self.seed,
                shuffle=False,
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

            if self.landmarks:
                # Mirror the dataset's landmark injection so the test input shape
                # matches training (the model expects the same N landmark nodes).
                T = seq.shape[0]
                L = len(self.landmarks)
                raw = torch.tensor([[x, y] for x, y, _ in self.landmarks],
                                   dtype=torch.float32)
                norm = (raw - self.mu) / self.sigma
                team_ids = torch.tensor([t for _, _, t in self.landmarks],
                                        dtype=seq.dtype)
                land_nodes = torch.zeros((T, L, 6), dtype=seq.dtype)
                land_nodes[:, :, :2] = norm
                land_nodes[:, :, 4] = 0.0
                land_nodes[:, :, 5] = team_ids
                seq = torch.cat([seq, land_nodes], dim=1)

            traj = model.get_trajectory(seq, self.mu, self.sigma)
            # traj = traj[8:, :, :2].reshape(-1)
            traj = traj[8:, :11, :2].reshape(-1)
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
    import argparse
    from lightning.pytorch.callbacks import ModelCheckpoint

    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="eqmotion")
    # tuned-best defaults (optuna #117): hidden_nf=64, hid_channel=64, n_layers=2
    p.add_argument("--hidden-nf", type=int, default=64)
    p.add_argument("--hid-channel", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1.8e-3)
    p.add_argument("--weight-decay", type=float, default=2e-6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=0.66)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument(
        "--lr-scheduler", default="cosine", choices=["none", "cosine", "plateau"]
    )
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--iso-norm", action="store_true")
    p.add_argument(
        "--add-hoops",
        action="store_true",
        help="Inject 2 static basket nodes (court-frame / D2 structure).",
    )
    p.add_argument(
        "--full-val",
        action="store_true",
        help="Deterministic multi-window validation for a stable val/mse_ft.",
    )
    p.add_argument(
        "--val-windows",
        type=int,
        default=8,
        help="Windows per sequence for --full-val.",
    )
    p.add_argument(
        "--landmarks",
        default="",
        help="Comma-separated landmark preset names from LANDMARK_SETS "
             "(e.g. 'hoops,ft,3pt'). Overrides --add-hoops if given.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-submit", action="store_true")
    args = p.parse_args()

    L.seed_everything(args.seed, workers=True)

    # Resolve landmarks: explicit --landmarks wins; else fall back to --add-hoops.
    if args.landmarks:
        landmarks = landmarks_from_spec(args.landmarks)
    elif args.add_hoops:
        landmarks = LANDMARK_SETS["hoops"]
    else:
        landmarks = []
    print(f"[{args.run_name}] landmarks={args.landmarks or ('hoops' if args.add_hoops else 'none')} "
          f"(n={len(landmarks)})")

    data_module = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=args.batch_size,
        seed=args.seed,
        iso_norm=args.iso_norm,
        landmarks=landmarks,
        full_val=args.full_val,
        val_windows_per_seq=args.val_windows,
    )

    model = NBAEqMotionLightningModel(
        hidden_nf=args.hidden_nf,
        hid_channel=args.hid_channel,
        n_layers=args.n_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        n_landmarks=len(landmarks),
    )

    wandb_logger = WandbLogger(project="NML_base", name=args.run_name)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(PROJECT_ROOT / "checkpoints" / "eqmotion" / args.run_name),
        filename="best",
        monitor="val/mse_ft",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val/mse_ft", patience=args.patience, mode="min")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        accelerator="auto",
        gradient_clip_val=args.grad_clip,
        callbacks=[ckpt_cb, early_stop],
    )

    trainer.fit(model, data_module)
    print(f"[{args.run_name}] best val/mse_ft = {ckpt_cb.best_model_score.item():.4f}")

    if not args.no_submit:
        # Submit from the BEST checkpoint, not the final-epoch model in memory
        # (final epoch is typically worse than the checkpointed minimum).
        best = (
            NBAEqMotionLightningModel.load_from_checkpoint(
                ckpt_cb.best_model_path, strict=False
            )
            .to(model.device)
            .eval()
        )
        data_module.get_kaggle_submission(best, str(TEST_DIR), str(SUBMISSION_DIR))
        print(f"[{args.run_name}] submission written from {ckpt_cb.best_model_path}")
