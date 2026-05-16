"""
HHT-CFI adapted for NBA trajectory prediction.

Format bridge:
  NBA input:  X [B, T_obs=8, N=11, 6], y [B, T_pred=12, N=11, 6]  (normalized)
  HHT-CFI:   (batch_abs_gt [H, B*N, 3], batch_norm_gt [H, B*N, 2],
               batch_split, shift_values [1, B*N, 2], max_values [B*N, 2])
              edge_pair {(left, right): [Tensor [N*(N-1), 2]]}   local indices

Normalization strategy:
  Global mu/sigma (NBA-style) → absolute feet → HHT-CFI per-agent normalization
  shift_values = last observed absolute position per agent
  max_values   = max |centered obs| per agent, clamped to ≥1 ft

The social-interaction threshold in the decoder (10 ft) operates in absolute
foot-space, which is correct for NBA since the court is ~94 x 50 ft.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L
import pandas as pd
import dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).resolve().parents[1]          # .../src/
_ROOT = Path(__file__).resolve().parents[2]          # .../network_ml_project/

sys.path.insert(0, str(_SRC))   # utils.metrics + hht_cfi package

from hht_cfi.models import MyTraj  # noqa: E402
from utils.metrics import compute_ade, compute_fde, compute_mse  # noqa: E402

dotenv.load_dotenv(dotenv.find_dotenv())

DATA_DIR  = _ROOT / "data"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR  = DATA_DIR / "test"  / "test"
SUB_DIR   = _ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)


# ── HHT-CFI args ───────────────────────────────────────────────────────────────
class _Args:
    hidden_size      = 64
    obs_length       = 8
    pred_length      = 12
    seq_length       = 20
    final_mode       = 20
    input_offset     = True   # use velocity offsets as encoder input
    input_mix        = False
    input_position   = False
    x_encoder_layers = 3
    x_encoder_head   = 8

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ── Data pipeline ──────────────────────────────────────────────────────────────
class NBADataset(Dataset):
    def __init__(self, files, context_size, horizon_size, mu, sigma):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size  = context_size + horizon_size
        self._load(files, mu, sigma)

    def _load(self, files, mu, sigma):
        self.sequences = []
        self.max_start = []
        for f in files:
            seq = torch.load(f, weights_only=False)
            seq[:, :, :2] = (seq[:, :, :2] - mu) / sigma
            vel = torch.zeros_like(seq[:, :, :2])
            vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
            seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)
            self.sequences.append(seq)
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_idx, start = index
        X = self.sequences[seq_idx][start : start + self.context_size]
        y = self.sequences[seq_idx][start + self.context_size : start + self.window_size]
        return X, y


class NBASampler(Sampler):
    def __init__(self, batch_size, max_start, seed=0, shuffle=True):
        self.batch_size = batch_size
        self.max_start  = max_start
        self.seed       = seed
        self.shuffle    = shuffle
        self.epoch      = 0
        self.generator  = torch.Generator().manual_seed(seed)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __len__(self):
        return len(self.max_start)

    def __iter__(self):
        n    = len(self)
        perm = torch.randperm(n, generator=self.generator).tolist() if self.shuffle else list(range(n))
        perm_start = [(i, self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            for idx, max_start in perm_start[k : k + self.batch_size]:
                start = torch.randint(0, max_start + 1, size=(), generator=self.generator)
                yield idx, start


class NBADataModule(L.LightningDataModule):
    def __init__(self, split_path, batch_size=16, context_size=8, horizon_size=12, seed=0):
        super().__init__()
        self.split_path   = split_path
        self.batch_size   = batch_size
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.seed         = seed
        self.mu = self.sigma = None

    def setup(self, stage=None):
        manifest    = json.loads(Path(self.split_path).read_text())
        data_dir    = _ROOT / manifest["data_dir"]
        train_files = [data_dir / f for f in manifest["train"]]
        val_files   = [data_dir / f for f in manifest["val"]]
        self.mu, self.sigma = self._norm_stats(train_files)
        self.train_ds = NBADataset(train_files, self.context_size, self.horizon_size, self.mu, self.sigma)
        self.val_ds   = NBADataset(val_files,   self.context_size, self.horizon_size, self.mu, self.sigma)

    def _norm_stats(self, files):
        all_pos = torch.cat([torch.load(f, weights_only=False)[:, :, :2] for f in files])
        return all_pos.mean((0, 1)), all_pos.std((0, 1))

    def train_dataloader(self):
        s = NBASampler(self.batch_size, self.train_ds.max_start, seed=self.seed, shuffle=True)
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=s)

    def val_dataloader(self):
        s = NBASampler(self.batch_size, self.val_ds.max_start, seed=self.seed, shuffle=False)
        return DataLoader(self.val_ds, batch_size=self.batch_size, sampler=s)

    def get_kaggle_submission(self, model: "NBAHHTCFILightningModel", test_dir: str, target_dir: str):
        rows = []
        for f in sorted(os.listdir(test_dir)):
            if not f.endswith(".pt"):
                continue
            seq = torch.load(os.path.join(test_dir, f), weights_only=False)
            seq[:, :, :2] = (seq[:, :, :2] - self.mu) / self.sigma
            vel = torch.zeros_like(seq[:, :, :2])
            vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
            seq = torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)
            context = seq[-self.context_size:]               # last 8 frames
            traj = model.get_trajectory(context, self.mu, self.sigma)
            flat = traj[self.context_size:, :, :2].reshape(-1)
            rows.append([int(f.removesuffix(".pt"))] + flat.tolist())

        cols = ["id"] + [
            f"entity_{i}_time_{t}_{ax}"
            for t in range(self.horizon_size)
            for i in range(11)
            for ax in ["x", "y"]
        ]
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        df  = pd.DataFrame(rows, columns=cols).set_index("id").sort_index()
        out = os.path.join(target_dir, f"solution_hhtcfi_{ts}.csv")
        df.to_csv(out)
        print(f"Saved: {out}")


# ── Edge cache & format conversion ─────────────────────────────────────────────
_EDGE_CACHE: dict[int, Tensor] = {}


def _full_graph_edges(n: int, device) -> Tensor:
    """Directed complete graph on n nodes, LOCAL indices 0..n-1."""
    if n not in _EDGE_CACHE:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        _EDGE_CACHE[n] = torch.tensor(pairs, dtype=torch.long)
    return _EDGE_CACHE[n].to(device)


def _to_hht_inputs(X: Tensor, y: Tensor, mu: Tensor, sigma: Tensor):
    """
    Convert NBA batch to HHT-CFI forward inputs.

    X: [B, T_obs, N, 6]  — normalized (x, y, dx, dy, isplayer, team)
    y: [B, T_pred, N, 6] — normalized

    Returns (inputs, edge_pair) ready for MyTraj.forward().
    """
    B, T_obs, N, _ = X.shape
    T_pred = y.shape[1]
    H      = T_obs + T_pred
    dev    = X.device

    # 1. Denormalize to absolute feet
    abs_obs  = X[:, :, :, :2] * sigma + mu           # [B, T_obs,  N, 2]
    abs_pred = y[:, :, :, :2] * sigma + mu           # [B, T_pred, N, 2]
    abs_full = torch.cat([abs_obs, abs_pred], dim=1) # [B, H, N, 2]

    # 2. HHT-CFI per-agent normalization
    shift   = abs_obs[:, -1, :, :]                               # [B, N, 2]
    centered = abs_full - shift.unsqueeze(1)                     # [B, H, N, 2]
    max_v   = centered[:, :T_obs].abs().amax(dim=1).clamp(min=1.0)  # [B, N, 2]
    hht_norm = centered / max_v.unsqueeze(1)                    # [B, H, N, 2]

    # 3. Flatten: scene-major order [H, B*N, 2]
    norm_flat = hht_norm.permute(1, 0, 2, 3).reshape(H, B * N, 2)
    abs_flat  = abs_full.permute(1, 0, 2, 3).reshape(H, B * N, 2)

    # Agent type: ball=0, team_A=1, team_B=2
    # is_player ∈ {0,1}, team_id ∈ {-1,0,1} (-1=team_B, 0=ball, 1=team_A)
    agent_type = (X[:, 0, :, 4].long() + (X[:, 0, :, 5] < 0).long()).reshape(B * N)  # [B*N]
    agent_type_col = agent_type.float().unsqueeze(0).expand(H, -1).unsqueeze(-1)       # [H, B*N, 1]
    batch_abs_gt = torch.cat([abs_flat, agent_type_col], dim=-1)

    # batch_split: one (left, right) tensor pair per scene
    batch_split = [(torch.tensor(i * N), torch.tensor((i + 1) * N)) for i in range(B)]

    # shift_values [1, B*N, 2]: decoder does squeeze(0) → [B*N, 2]
    shift_flat = shift.reshape(1, B * N, 2)
    max_flat   = max_v.reshape(B * N, 2)

    inputs = (batch_abs_gt, norm_flat, batch_split, shift_flat, max_flat)

    # edge_pair: full directed graph with LOCAL indices 0..N-1 per scene
    local_edges = _full_graph_edges(N, dev)
    edge_pair   = {(i * N, (i + 1) * N): [local_edges] for i in range(B)}

    return inputs, edge_pair


# ── Inference toggles ───────────────────────────────────────────────────────────
# To disable an improvement: set its flag to False, or comment out the True line.
_USE_MIN_SCALE = True   # pick most confident mode (min total output scale) — no dummy_y needed
_USE_TTA       = True   # y-flip test-time augmentation (2× forward passes, averaged)
_USE_CLAMP     = True  # clip predictions to NBA court bounds

_COURT_HALF_LEN = 47.5     # ft  (x-axis: baseline to baseline)
_COURT_HALF_WID = 25.0     # ft  (y-axis: sideline to sideline)


# ── Model wrapper ───────────────────────────────────────────────────────────────
class NBAHHTCFIModel(nn.Module):
    """Wraps MyTraj for batched NBA sequences."""

    def __init__(self, hidden_size=64, x_encoder_layers=3, x_encoder_head=8):
        super().__init__()
        self.args = _Args(
            hidden_size=hidden_size,
            x_encoder_layers=x_encoder_layers,
            x_encoder_head=x_encoder_head,
        )
        self.model  = MyTraj(self.args)
        self.T_obs  = self.args.obs_length
        self.T_pred = self.args.pred_length

    def forward(self, X: Tensor, y: Tensor, mu: Tensor, sigma: Tensor, epoch: int = 0):
        """
        Training forward: returns (combined_loss, full_pre_tra).
        full_pre_tra[0]: [T_obs-1 + T_pred, B*N, 2]  HHT-CFI normalized space.
        """
        inputs, edge_pair = _to_hht_inputs(X, y, mu, sigma)
        (loss1, loss2), full_pre_tra = self.model(inputs, edge_pair, epoch)
        return loss1 + loss2, full_pre_tra

    def _forward_single(self, X_obs: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """Single forward pass → denormalized absolute positions [T_pred, B*N, 2]."""
        B, _, N, _ = X_obs.shape
        # Dummy future: last-frame CV (only needed for the training loss path; mode
        # selection at inference uses scale-based criterion, not this target).
        vel_norm = X_obs[:, -1, :, 2:4]
        pos_norm = X_obs[:, -1, :, :2]
        t = torch.arange(1, self.T_pred + 1, dtype=X_obs.dtype, device=X_obs.device)
        future_pos = pos_norm.unsqueeze(1) + vel_norm.unsqueeze(1) * t.view(1, -1, 1, 1)
        dummy_y = X_obs[:, -1:, :, :].expand(-1, self.T_pred, -1, -1).clone()
        dummy_y[:, :, :, :2] = future_pos

        inputs, edge_pair = _to_hht_inputs(X_obs, dummy_y, mu, sigma)
        _, full_pre_tra = self.model(inputs, edge_pair, epoch=0)

        if _USE_MIN_SCALE:
            # Pick the mode the model is most confident about (min total output scale).
            # Independent of dummy_y — uses the model's own uncertainty estimate.
            out_mu_all    = full_pre_tra[2]   # [K=20, B*N, T_pred, 2]
            out_sigma_all = full_pre_tra[3]   # [K=20, B*N, T_pred, 2]
            BN = out_mu_all.shape[1]
            best = out_sigma_all.sum(dim=(-1, -2)).argmin(dim=0)  # [B*N]
            pred_hht = out_mu_all[best, torch.arange(BN)].permute(1, 0, 2)  # [T_pred, B*N, 2]
        else:
            pred_hht = full_pre_tra[0][-self.T_pred:]  # ADE-optimal mode by CV dummy

        abs_obs = X_obs[:, :, :, :2] * sigma + mu
        shift   = abs_obs[:, -1].reshape(B * N, 2)
        max_v   = (abs_obs - abs_obs[:, -1:]).abs().amax(1).reshape(B * N, 2).clamp(min=1.0)
        return pred_hht * max_v.unsqueeze(0) + shift.unsqueeze(0)

    def predict(self, X_obs: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Inference without ground truth.

        X_obs: [B, T_obs, N, 6]  normalized input.
        Returns: [T_pred, B*N, 2]  absolute positions in feet.

        Toggle improvements via module-level flags above the class definition:
          _USE_MIN_SCALE, _USE_TTA, _USE_CLAMP
        """
        pred = self._forward_single(X_obs, mu, sigma)  # [T_pred, B*N, 2]

        if _USE_TTA:
            # Mirror the scene across the court's y-axis (sideline symmetry).
            # Negate normalized y and dy, run a second forward pass, flip back, average.
            X_flip = X_obs.clone()
            X_flip[:, :, :, 1] = -X_obs[:, :, :, 1]   # negate normalized y position
            X_flip[:, :, :, 3] = -X_obs[:, :, :, 3]   # negate normalized y velocity (dy)
            pred_flip = self._forward_single(X_flip, mu, sigma)
            pred_flip[:, :, 1] = -pred_flip[:, :, 1]   # flip predicted y back
            pred = (pred + pred_flip) / 2

        if _USE_CLAMP:
            pred[:, :, 0].clamp_(-_COURT_HALF_LEN, _COURT_HALF_LEN)
            pred[:, :, 1].clamp_(-_COURT_HALF_WID, _COURT_HALF_WID)

        return pred

    def all_modes(self, X_obs: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """All K decoder modes → [K, T_pred, B*N, 2] denormalized absolute positions."""
        B, _, N, _ = X_obs.shape
        vel_norm = X_obs[:, -1, :, 2:4]
        pos_norm = X_obs[:, -1, :, :2]
        t = torch.arange(1, self.T_pred + 1, dtype=X_obs.dtype, device=X_obs.device)
        future_pos = pos_norm.unsqueeze(1) + vel_norm.unsqueeze(1) * t.view(1, -1, 1, 1)
        dummy_y = X_obs[:, -1:, :, :].expand(-1, self.T_pred, -1, -1).clone()
        dummy_y[:, :, :, :2] = future_pos

        inputs, edge_pair = _to_hht_inputs(X_obs, dummy_y, mu, sigma)
        _, full_pre_tra = self.model(inputs, edge_pair, epoch=0)

        out_mu_all = full_pre_tra[2]  # [K=20, B*N, T_pred, 2] in HHT space

        abs_obs = X_obs[:, :, :, :2] * sigma + mu
        shift   = abs_obs[:, -1].reshape(B * N, 2)                                     # [B*N, 2]
        max_v   = (abs_obs - abs_obs[:, -1:]).abs().amax(1).reshape(B * N, 2).clamp(min=1.0)

        # denormalize: [K, B*N, T_pred, 2] → [K, T_pred, B*N, 2]
        denorm = out_mu_all * max_v.unsqueeze(0).unsqueeze(2) + shift.unsqueeze(0).unsqueeze(2)
        return denorm.permute(0, 2, 1, 3)


# ── Lightning module ────────────────────────────────────────────────────────────
class NBAHHTCFILightningModel(L.LightningModule):
    def __init__(
        self,
        hidden_size: int      = 64,
        x_encoder_layers: int = 3,
        x_encoder_head: int   = 8,
        lr: float             = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = NBAHHTCFIModel(hidden_size, x_encoder_layers, x_encoder_head)
        self.mu: Tensor | None    = None
        self.sigma: Tensor | None = None

    def on_fit_start(self):
        dm = self.trainer.datamodule
        # Plain attributes (not buffers) so mu/sigma don't pollute the checkpoint
        # state dict.  The notebook reattaches them from the datamodule after load.
        self.mu    = dm.mu.to(self.device)
        self.sigma = dm.sigma.to(self.device)

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss, _ = self.net(X, y, self.mu, self.sigma, epoch=self.current_epoch)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        B, T_pred, N = y.shape[0], y.shape[1], y.shape[2]
        loss, full_pre_tra = self.net(X, y, self.mu, self.sigma, epoch=self.current_epoch)

        # pred_hht: [T_pred, B*N, 2] in HHT-CFI normalized space
        pred_hht = full_pre_tra[0][-T_pred:]

        # denormalize to absolute feet
        abs_obs  = X[:, :, :, :2] * self.sigma + self.mu
        shift    = abs_obs[:, -1].reshape(B * N, 2)
        max_v    = (abs_obs - abs_obs[:, -1:]).abs().amax(1).reshape(B * N, 2).clamp(min=1.0)
        pred_abs = pred_hht * max_v.unsqueeze(0) + shift.unsqueeze(0)   # [T_pred, B*N, 2]

        tgt_abs  = (y[:, :, :, :2] * self.sigma + self.mu).permute(1, 0, 2, 3).reshape(T_pred, B * N, 2)

        self.log("val/loss",   loss,                            on_epoch=True, prog_bar=True)
        self.log("val/ade_ft", compute_ade(pred_abs, tgt_abs),  on_epoch=True, prog_bar=True)
        self.log("val/fde_ft", compute_fde(pred_abs, tgt_abs),  on_epoch=True, prog_bar=True)
        self.log("val/mse_ft", compute_mse(pred_abs, tgt_abs),  on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": sched}

    def predict_batch(self, X_batch: Tensor) -> Tensor:
        """
        X_batch: [B, T_obs, N, 6] — normalized input (mu/sigma already attached).
        Returns: pred_abs [T_pred, B*N, 2] in absolute feet, using CV dummy target.
        """
        return self.net.predict(X_batch, self.mu, self.sigma)

    def predict_batch_all_modes(self, X_batch: Tensor) -> Tensor:
        """
        X_batch: [B, T_obs, N, 6] — normalized input.
        Returns: [K=20, T_pred, B*N, 2] — all decoder modes in absolute feet.
        """
        return self.net.all_modes(X_batch, self.mu, self.sigma)

    def get_trajectory(self, X_context: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        X_context: [T_obs, N, 6]  — normalized, last T_obs frames of a sequence.
        Returns:   [T_obs + T_pred, N, 4]  — absolute coords + is_player + team_id.
        """
        self.eval()
        N = X_context.shape[1]
        with torch.no_grad():
            pred_abs = self.net.predict(
                X_context.unsqueeze(0).to(self.device),
                mu.to(self.device), sigma.to(self.device),
            ).cpu()                                                 # [T_pred, N, 2]

        obs_abs  = (X_context[:, :, :2] * sigma + mu).cpu()       # [T_obs, N, 2]
        static   = X_context[-1, :, 4:].unsqueeze(0).expand(self.net.T_pred, -1, -1).cpu()
        pred_out = torch.cat([pred_abs, static], dim=-1)           # [T_pred, N, 4]
        obs_out  = torch.cat([obs_abs,  X_context[:, :, 4:].cpu()], dim=-1)   # [T_obs, N, 4]
        return torch.cat([obs_out, pred_out], dim=0)               # [T_obs+T_pred, N, 4]


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import WandbLogger

    L.seed_everything(0)

    dm = NBADataModule(
        split_path=str(_ROOT / "splits" / "fold0.json"),
        batch_size=16,   # smaller than EqMotion — decoder builds [B*N, B*N] distance matrix
    )

    model = NBAHHTCFILightningModel(lr=1e-3)

    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        logger=WandbLogger(project="NML_base", name="hht_cfi"),
        callbacks=[EarlyStopping(monitor="val/loss", patience=25, mode="min")],
    )

    trainer.fit(model, dm)
    dm.get_kaggle_submission(model, str(TEST_DIR), str(SUB_DIR))
