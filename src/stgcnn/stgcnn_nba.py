"""
Social-STGCNN for NBA trajectory prediction — faithful reference baseline.

This is a faithful port of the official Social-STGCNN pipeline
(https://github.com/abduallahmohamed/Social-STGCNN) to our NBA Lightning + W&B
setup, kept deliberately close to the reference so it can serve as a starting
point before any custom changes. The reference's key design choices are
reproduced here (see src/stgcnn/ref_code/):

  * Graph (ref_code/utils.py: anorm + seq_to_graph):
      per observed frame, the adjacency is reciprocal distance 1/‖·‖ computed in
      *relative* (velocity / displacement) space, with a unit self-loop, then
      converted to the normalized graph Laplacian L = I − D^{-1/2} A D^{-1/2}.
  * Targets (ref_code/test.py: nodes_rel_to_nodes_abs):
      the model predicts per-step *displacements*; absolute positions are
      recovered by cumulative-summing from the last observed position.
  * Backbone (ref_code/model.py): the official `social_stgcnn` — one ST-GCN
      layer (2→5 channels) followed by `n_txpcnn` temporal-extrapolation convs.
      The graph-conv einsum is the only change: it now carries a batch index so
      we can train with batched, per-sample graphs instead of batch_size=1.
  * Loss (ref_code/metrics.py: bivariate_loss): per-step bivariate-Gaussian NLL
      over displacements (σ via exp, ρ via tanh applied inside the loss).
  * No coordinate normalization — the reference works directly in raw units; in
      displacement space the values are already small, so we keep feet.

The one deliberate deviation is the sampler: we keep the project's
NBASampler (one random window per sequence per epoch) instead of enumerating
every sliding window, so runs stay comparable to the EqMotion pipeline.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
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

dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = DATA_DIR / "test" / "test"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

N_ENTITIES = 11  # 10 players + ball


# ── Model (faithful port of ref_code/model.py, batched-einsum fix) ─────────────


class ConvTemporalGraphical(nn.Module):
    """Multi-relation graph convolution.

    `kernel_size` is the number of relation types K (e.g. position + velocity).
    A 1×1 conv produces K separate feature blocks; each is aggregated with its
    own relation graph and the results are summed — the standard ST-GCN spatial
    partition, here repurposed for relation types. A carries a batch axis so each
    sample uses its own per-frame graphs.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size  # number of relations K
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))

    def forward(self, x, A):
        # x: [B, C_in, T, V]   A: [B, K, T, V, V]
        x = self.conv(x)
        B, _, T, V = x.shape
        x = x.view(B, self.kernel_size, self.out_channels, T, V)
        x = torch.einsum("nkctv,nktvw->nctw", (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    """Spatial-temporal graph conv block: graph conv + temporal conv + residual."""

    def __init__(
        self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1,
        dropout=0, residual=True,
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        if not self.use_mdn:
            x = self.prelu(x)
        return x, A


class social_stgcnn(nn.Module):
    def __init__(
        self, n_stgcnn=1, n_txpcnn=1, input_feat=2, hidden_feat=5, output_feat=5,
        seq_len=8, pred_seq_len=12, kernel_size=3, n_relations=1,
    ):
        super().__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        # ST-GCN stack runs at hidden_feat width (reference tied this to the
        # 5-param output, leaving no hidden representation); a 1×1 conv at the
        # very end projects hidden_feat -> output_feat (the Gaussian params).
        # The spatial kernel is the number of relation types (n_relations).
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, hidden_feat, (kernel_size, n_relations)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(hidden_feat, hidden_feat, (kernel_size, n_relations)))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for _ in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(self.n_txpcnn)])
        self.output_proj = nn.Conv2d(hidden_feat, output_feat, 1)

    def forward(self, v, a):
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)  # [B, hidden, obs, N]

        # Reference's view-based reshuffle of (channels ↔ time): [B,C,T,V] -> [B,T,C,V]
        # so the TXP convs treat the time axis as channels. (A reshape, not a
        # transpose — kept exactly as the reference does it.)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))
        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])  # [B, hidden, pred, N]
        v = self.output_proj(v)  # [B, output_feat, pred, N]
        return v, a


class GraphGRUDecoder(nn.Module):
    """Autoregressive rollout with a live position-space graph.

    At each future step: (1) rebuild the proximity graph from the current
    predicted positions, (2) aggregate neighbour hidden states over it,
    (3) update a per-node GRU cell, (4) emit the next displacement, (5) step
    the position forward and feed it back. Each step is conditioned on the
    previous one (the fix for one-shot drift) and interactions stay live as the
    play evolves. The graph is built from detached positions for stability —
    gradients still flow through the GRU and the displacement feedback.
    """

    def __init__(self, hidden, pred_len, output_feat=5):
        super().__init__()
        self.pred_len = pred_len
        self.hidden = hidden
        self.disp_embed = nn.Linear(2, hidden)
        self.graph_lin = nn.Linear(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, output_feat)

    def forward(self, h0, pos0, vel0):
        # h0 [B,N,H]  pos0 [B,N,2]  vel0 [B,N,2]
        B, N, H = h0.shape
        h, pos, d_prev = h0, pos0, vel0
        outs = []
        for _ in range(self.pred_len):
            A = _laplacian_from(pos.detach())  # [B, N, N]
            msg = torch.einsum("bnm,bmh->bnh", A, self.graph_lin(h))  # neighbour mix
            inp = msg + self.disp_embed(d_prev)  # [B, N, H]
            h = self.gru(inp.reshape(B * N, H), h.reshape(B * N, H)).reshape(B, N, H)
            out = self.head(h)  # [B, N, output_feat]
            outs.append(out)
            d = out[..., :2]  # mean displacement
            pos = pos + d
            d_prev = d
        return torch.stack(outs, dim=1)  # [B, pred, N, output_feat]


class stgcnn_autoreg(nn.Module):
    """ST-GCN encoder over the observed window + autoregressive graph decoder."""

    def __init__(
        self, n_stgcnn=2, input_feat=4, hidden_feat=64, output_feat=5,
        seq_len=8, pred_seq_len=12, kernel_size=3, n_relations=1,
    ):
        super().__init__()
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, hidden_feat, (kernel_size, n_relations)))
        for _ in range(1, n_stgcnn):
            self.st_gcns.append(st_gcn(hidden_feat, hidden_feat, (kernel_size, n_relations)))
        self.decoder = GraphGRUDecoder(hidden_feat, pred_seq_len, output_feat)

    def forward(self, X, A, last_pos):
        # X [B, C, obs, N]   A [B, R, obs, N, N]   last_pos [B, N, 2]
        v = X
        for gcn in self.st_gcns:
            v, A = gcn(v, A)  # [B, hidden, obs, N]
        h0 = v[:, :, -1, :].permute(0, 2, 1)  # [B, N, hidden] — last observed frame
        vel0 = X[:, :2, -1, :].permute(0, 2, 1)  # [B, N, 2] — last observed velocity
        return self.decoder(h0, last_pos, vel0)  # [B, pred, N, output_feat]


# ── Graph + loss helpers (faithful to ref_code/utils.py & metrics.py) ──────────


def _laplacian_from(coords: Tensor) -> Tensor:
    """Per-frame normalized Laplacian from reciprocal distances in `coords`.

    coords: [T, N, 2] -> [T, N, N]. Edge weight = 1/‖·‖ (0 if coincident),
    unit self-loop, then L = I − D^{-1/2} A D^{-1/2} (anorm + normalized Laplacian).
    """
    T, N, _ = coords.shape
    dev = coords.device
    dist = torch.cdist(coords, coords)  # [T, N, N]
    A = torch.where(dist > 0, 1.0 / dist, torch.zeros_like(dist))
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    A[:, eye] = 1.0  # unit self-loops
    deg = A.sum(-1)  # [T, N]
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv = torch.diag_embed(d_inv_sqrt)  # [T, N, N]
    eye_T = torch.eye(N, device=dev).expand(T, N, N)
    return eye_T - D_inv @ A @ D_inv


def build_graph(abs_obs: Tensor, rel_obs: Tensor, space: str) -> Tensor:
    """Stack per-relation graphs -> [R, T, N, N].

    space: "pos" → proximity in absolute position (who is near whom — the ball
    connects to its nearby handler), "vel" → similarity in velocity (the original
    reference graph), "both" → two relation channels the model weights separately.
    """
    rels = []
    if space in ("pos", "both"):
        rels.append(_laplacian_from(abs_obs))
    if space in ("vel", "both"):
        rels.append(_laplacian_from(rel_obs))
    return torch.stack(rels, dim=0)  # [R, T, N, N]


def n_relations_for(space: str) -> int:
    return 2 if space == "both" else 1


def bivariate_loss(V_pred: Tensor, V_trgt: Tensor) -> Tensor:
    """Bivariate-Gaussian NLL over displacements (ref_code/metrics.py).

    V_pred: [..., 5] raw params [mux, muy, log σx, log σy, ρ-pre-tanh]
    V_trgt: [..., 2] target displacements
    """
    normx = V_trgt[..., 0] - V_pred[..., 0]
    normy = V_trgt[..., 1] - V_pred[..., 1]
    sx = torch.exp(V_pred[..., 2])
    sy = torch.exp(V_pred[..., 3])
    corr = torch.tanh(V_pred[..., 4])

    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    result = torch.exp(-z / (2 * negRho))
    denom = 2 * torch.pi * (sxsy * torch.sqrt(negRho))
    result = result / denom
    result = -torch.log(torch.clamp(result, min=1e-20))
    return torch.mean(result)


def gaussian_params(V_pred: Tensor):
    """Split raw model output [..., 5] into (mean[...,2], sx, sy, corr)."""
    mean = V_pred[..., 0:2]
    sx = torch.exp(V_pred[..., 2])
    sy = torch.exp(V_pred[..., 3])
    corr = torch.tanh(V_pred[..., 4])
    return mean, sx, sy, corr


def sample_displacements(V_pred: Tensor, k: int) -> Tensor:
    """Draw K displacement samples from the per-step bivariate Gaussian.

    V_pred: [B, T, N, 5] -> samples [K, B, T, N, 2] via the 2×2 Cholesky factor.
    """
    mean, sx, sy, corr = gaussian_params(V_pred)
    l11 = sx
    l21 = corr * sy
    l22 = sy * torch.sqrt(torch.clamp(1 - corr ** 2, min=1e-6))
    eps = torch.randn((k,) + mean.shape, device=V_pred.device)  # [K, B, T, N, 2]
    s_x = l11 * eps[..., 0]
    s_y = l21 * eps[..., 0] + l22 * eps[..., 1]
    return mean.unsqueeze(0) + torch.stack([s_x, s_y], dim=-1)


def rel_to_abs(disp: Tensor, last_pos: Tensor, time_dim: int) -> Tensor:
    """Integrate displacements to absolute positions from the last observed pos."""
    return disp.cumsum(dim=time_dim) + last_pos.unsqueeze(time_dim)


# ── Data ───────────────────────────────────────────────────────────────────────


class NBADataset(Dataset):
    """Yields windows as (X, A, V_tr, last_pos, abs_target).

    X         [2, obs, N]   relative (displacement) features, channel-first
    A         [obs, N, N]   per-frame normalized-Laplacian graph (rel space)
    V_tr      [pred, N, 2]  target displacements (NLL target)
    last_pos  [N, 2]        last observed absolute position (integration anchor)
    abs_target[pred, N, 2]  absolute future positions (metric ground truth)
    """

    def __init__(self, files, context_size, horizon_size, graph_space="pos"):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.graph_space = graph_space
        self.load_data(files)

    def load_data(self, files):
        self.sequences = []  # full features [T, N, 4]: x, y, isplayer, team
        self.max_start = []
        for f in files:
            seq = torch.load(f, weights_only=False).float()
            self.sequences.append(seq[:, :, :4])
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index):
        seq_idx, start = index
        c = self.context_size
        win = self.sequences[seq_idx][start : start + self.window_size]  # [W, N, 4]
        abs_win = win[:, :, :2]
        static = win[c - 1, :, 2:4]  # [N, 2] isplayer, team — static over time

        rel = torch.zeros_like(abs_win)
        rel[1:] = abs_win[1:] - abs_win[:-1]  # displacement; rel[0]=0

        rel_obs = rel[:c]  # [obs, N, 2]
        V_tr = rel[c:]  # [pred, N, 2]
        A = build_graph(abs_win[:c], rel_obs, self.graph_space)  # [R, obs, N, N]
        # Node features [dx, dy, isplayer, team]; static feats broadcast over time.
        static_obs = static.unsqueeze(0).expand(c, -1, -1)  # [obs, N, 2]
        feat = torch.cat([rel_obs, static_obs], dim=-1)  # [obs, N, 4]
        X = feat.permute(2, 0, 1)  # [4, obs, N]
        last_pos = abs_win[c - 1]  # [N, 2]
        abs_target = abs_win[c:]  # [pred, N, 2]
        return X, A, V_tr, last_pos, abs_target

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
                start = torch.randint(0, max_start + 1, size=(), generator=self.generator)
                yield idx, start

    def __len__(self):
        return len(self.max_start)


# ── Lightning module ────────────────────────────────────────────────────────────


class NBASTGCNNLightningModel(L.LightningModule):
    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        n_stgcnn: int = 2,
        n_txpcnn: int = 5,
        hidden_feat: int = 64,
        kernel_size: int = 3,
        graph_space: str = "pos",  # "pos" | "vel" | "both" — must match datamodule
        use_identity: bool = True,  # feed [isplayer, team] as node features
        optimizer: str = "adam",  # "adam" | "sgd"
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_samples: int = 20,
        loss_mode: str = "mse",   # "mse" | "nll" | "nll+mse"
        mse_weight: float = 1.0,  # λ on the MSE term in "nll+mse"
        decoder: str = "autoreg",  # "txp" (one-shot) | "autoreg" (graph rollout)
        augment: bool = True,  # random court-symmetry reflections (train only)
    ):
        assert loss_mode in ("mse", "nll", "nll+mse")
        assert optimizer in ("adam", "sgd")
        assert graph_space in ("pos", "vel", "both")
        assert decoder in ("txp", "autoreg")
        super().__init__()
        self.save_hyperparameters()
        common = dict(
            n_stgcnn=n_stgcnn,
            input_feat=4 if use_identity else 2,
            hidden_feat=hidden_feat,
            output_feat=5,
            seq_len=context_size,
            pred_seq_len=horizon_size,
            kernel_size=kernel_size,
            n_relations=n_relations_for(graph_space),
        )
        if decoder == "autoreg":
            self.net = stgcnn_autoreg(**common)
        else:
            self.net = social_stgcnn(n_txpcnn=n_txpcnn, **common)

    def forward(self, X: Tensor, A: Tensor, last_pos: Tensor) -> Tensor:
        # X [B,4,obs,N], A [B,R,obs,N,N], last_pos [B,N,2] -> V_pred [B, pred, N, 5]
        if not self.hparams.use_identity:
            X = X[:, :2]  # drop [isplayer, team] channels
        if self.hparams.decoder == "autoreg":
            return self.net(X, A, last_pos)  # [B, pred, N, 5]
        v, _ = self.net(X, A)  # [B, 5, pred, N]
        return v.permute(0, 2, 3, 1)

    def _abs_mean(self, V_pred: Tensor, last_pos: Tensor) -> Tensor:
        """Integrate the predicted mean displacement to absolute positions."""
        return rel_to_abs(V_pred[..., :2], last_pos, time_dim=1)  # [B, T, N, 2]

    def _losses(self, V_pred, V_tr, last_pos, abs_target):
        """Return (total_loss, components dict) per the configured loss_mode.

        MSE is computed on the *integrated absolute* trajectory, matching the
        Kaggle metric exactly (so early-step errors are penalized through drift).
        NLL is the reference bivariate-Gaussian loss on per-step displacements.
        """
        B = V_pred.shape[0]
        comps = {}
        if self.hparams.loss_mode in ("nll", "nll+mse"):
            comps["nll"] = bivariate_loss(V_pred.reshape(B, -1, 5), V_tr.reshape(B, -1, 2))
        if self.hparams.loss_mode in ("mse", "nll+mse"):
            abs_pred = self._abs_mean(V_pred, last_pos)
            comps["mse"] = ((abs_pred - abs_target) ** 2).mean()

        if self.hparams.loss_mode == "mse":
            total = comps["mse"]
        elif self.hparams.loss_mode == "nll":
            total = comps["nll"]
        else:
            total = comps["nll"] + self.hparams.mse_weight * comps["mse"]
        return total, comps

    def _augment(self, X, V_tr, last_pos, abs_target):
        """Random court-symmetry reflections, per instance (training only).

        The court is centered at the origin in raw feet, so a reflection is a
        coordinate negation. Reflections are isometries → pairwise distances are
        unchanged → the graph A is invariant and needs no flipping. Identity
        channels [isplayer, team] are unaffected. X is [B,4,obs,N] with channel
        0,1 = dx,dy; V_tr/abs_target are [B,*,N,2]; last_pos is [B,N,2].
        """
        B = X.shape[0]
        fx = torch.rand(B, device=X.device) < 0.5  # reflect across court's y-axis
        fy = torch.rand(B, device=X.device) < 0.5  # reflect across court's x-axis
        X, V_tr = X.clone(), V_tr.clone()
        last_pos, abs_target = last_pos.clone(), abs_target.clone()
        for mask, c in ((fx, 0), (fy, 1)):
            X[mask, c] = -X[mask, c]
            V_tr[mask, :, :, c] = -V_tr[mask, :, :, c]
            last_pos[mask, :, c] = -last_pos[mask, :, c]
            abs_target[mask, :, :, c] = -abs_target[mask, :, :, c]
        return X, V_tr, last_pos, abs_target

    def training_step(self, batch, batch_idx):
        X, A, V_tr, last_pos, abs_target = batch
        if self.hparams.augment:
            X, V_tr, last_pos, abs_target = self._augment(X, V_tr, last_pos, abs_target)
        V_pred = self(X, A, last_pos)
        loss, comps = self._losses(V_pred, V_tr, last_pos, abs_target)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        for name, val in comps.items():
            self.log(f"train/{name}", val, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, A, V_tr, last_pos, abs_target = batch
        V_pred = self(X, A, last_pos)  # [B, pred, N, 5]
        B, T, N, _ = V_pred.shape
        loss, _ = self._losses(V_pred, V_tr, last_pos, abs_target)

        # Mean-prediction metrics: integrate predicted mean displacement to abs.
        abs_pred = self._abs_mean(V_pred, last_pos)  # [B, T, N, 2]
        pred_flat = abs_pred.permute(1, 0, 2, 3).reshape(T, B * N, 2)
        target_flat = abs_target.permute(1, 0, 2, 3).reshape(T, B * N, 2)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/ade_ft", compute_ade(pred_flat, target_flat), on_epoch=True, prog_bar=True)
        self.log("val/fde_ft", compute_fde(pred_flat, target_flat), on_epoch=True, prog_bar=True)
        self.log("val/mse_ft", compute_mse(pred_flat, target_flat), on_epoch=True, prog_bar=True)

        # Best-of-K metrics only make sense when the Gaussian head is trained.
        if "nll" in self.hparams.loss_mode:
            k = self.hparams.n_samples
            disp_samples = sample_displacements(V_pred, k)  # [K, B, T, N, 2]
            abs_samples = rel_to_abs(disp_samples, last_pos.unsqueeze(0), time_dim=2)
            samples_flat = abs_samples.permute(0, 2, 1, 3, 4).reshape(k, T, B * N, 2)
            self.log(f"val/min_ade_ft_k{k}", compute_min_ade(samples_flat, target_flat), on_epoch=True)
            self.log(f"val/min_fde_ft_k{k}", compute_min_fde(samples_flat, target_flat), on_epoch=True)
            self.log(f"val/min_mse_ft_k{k}", compute_min_mse(samples_flat, target_flat), on_epoch=True)

        self._log_diagnostics(X, last_pos, abs_pred, abs_target)

    def _log_diagnostics(self, X, last_pos, abs_pred, abs_target):
        """Where does the error live? Ball-vs-players, per-step, and the
        constant-velocity baseline — logged so we can read them off W&B."""
        B, T, N, _ = abs_pred.shape
        sq = ((abs_pred - abs_target) ** 2).mean(-1)  # [B, T, N] (mean over x,y)

        # Ball vs players. Dataset packs node features [dx,dy,isplayer,team];
        # the ball is the entity with isplayer == 0.
        isplayer = X[:, 2, 0, :]  # [B, N] at first observed frame
        ball = (isplayer == 0).unsqueeze(1).expand(B, T, N)  # [B, T, N]
        self.log("val/mse_ball", sq[ball].mean(), on_epoch=True)
        self.log("val/mse_players", sq[~ball].mean(), on_epoch=True)

        # Error growth across the horizon (first / middle / last step).
        per_t = sq.mean(dim=(0, 2))  # [T]
        self.log("val/mse_t1", per_t[0], on_epoch=True)
        self.log("val/mse_tmid", per_t[T // 2], on_epoch=True)
        self.log("val/mse_tlast", per_t[-1], on_epoch=True)

        # Constant-velocity baseline: extrapolate the last observed velocity.
        last_vel = X[:, :2, -1, :].permute(0, 2, 1)  # [B, N, 2]
        steps = torch.arange(1, T + 1, device=X.device).view(1, T, 1, 1)
        abs_cv = last_pos.unsqueeze(1) + steps * last_vel.unsqueeze(1)  # [B, T, N, 2]
        self.log("val/cv_mse", ((abs_cv - abs_target) ** 2).mean(), on_epoch=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        return torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    @torch.no_grad()
    def predict_abs_mean(self, X: Tensor, A: Tensor, last_pos: Tensor) -> Tensor:
        """Absolute mean trajectory for a batch → [B, pred, N, 2]."""
        self.eval()
        last_pos = last_pos.to(self.device)
        V_pred = self(X.to(self.device), A.to(self.device), last_pos)
        mean_disp = V_pred[..., :2]
        return rel_to_abs(mean_disp, last_pos, time_dim=1).cpu()


# ── DataModule ──────────────────────────────────────────────────────────────────


class NBADataModule(L.LightningDataModule):
    def __init__(self, split_path, batch_size=128, context_size=8, horizon_size=12,
                 seed=0, graph_space="pos"):
        super().__init__()
        self.split_path = split_path
        self.batch_size = batch_size
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.seed = seed
        self.graph_space = graph_space

    def setup(self, stage=None):
        manifest = json.loads(Path(self.split_path).read_text())
        data_dir = PROJECT_ROOT / manifest["data_dir"]
        train_files = [data_dir / f for f in manifest["train"]]
        val_files = [data_dir / f for f in manifest["val"]]
        self.train_dataset = NBADataset(
            train_files, self.context_size, self.horizon_size, self.graph_space
        )
        self.val_dataset = NBADataset(
            val_files, self.context_size, self.horizon_size, self.graph_space
        )

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

    def get_kaggle_submission(self, model, test_dir: str, target_dir: str):
        c = self.context_size
        all_traj = []
        for f in sorted(os.listdir(test_dir)):
            if not f.endswith(".pt"):
                continue
            seq = torch.load(os.path.join(test_dir, f), weights_only=False).float()
            abs_obs = seq[:, :, :2]  # [obs, N, 2]
            static = seq[c - 1, :, 2:4]  # [N, 2] isplayer, team
            rel_obs = torch.zeros_like(abs_obs)
            rel_obs[1:] = abs_obs[1:] - abs_obs[:-1]

            A = build_graph(abs_obs, rel_obs, self.graph_space).unsqueeze(0)  # [1, R, obs, N, N]
            static_obs = static.unsqueeze(0).expand(rel_obs.shape[0], -1, -1)  # [obs, N, 2]
            feat = torch.cat([rel_obs, static_obs], dim=-1)  # [obs, N, 4]
            X = feat.permute(2, 0, 1).unsqueeze(0)  # [1, 4, obs, N]
            last_pos = abs_obs[c - 1].unsqueeze(0)  # [1, N, 2]

            abs_pred = model.predict_abs_mean(X, A, last_pos)[0]  # [pred, N, 2]
            traj = abs_pred[:, :N_ENTITIES, :2].reshape(-1)
            all_traj.append([int(f.removesuffix(".pt"))] + traj.tolist())

        df = (
            pd.DataFrame(
                all_traj,
                columns=["id"]
                + [
                    f"entity_{i}_time_{t}_{axis}"
                    for t in range(12)
                    for i in range(N_ENTITIES)
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

    GRAPH_SPACE = "pos"  # "pos" | "vel" | "both" — keep model & datamodule in sync

    data_module = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=128,
        graph_space=GRAPH_SPACE,
    )

    model = NBASTGCNNLightningModel(
        loss_mode="mse", optimizer="adam", graph_space=GRAPH_SPACE, decoder="autoreg"
    )

    h = model.hparams
    wandb_logger = WandbLogger(
        project="NML_base",
        name=f"stgcnn_{h.decoder}_{h.loss_mode}_{h.optimizer}_h{h.hidden_feat}"
        + f"_st{h.n_stgcnn}_{h.graph_space}"
        + ("_id" if h.use_identity else "")
        + ("_aug" if h.augment else ""),
    )
    checkpoint = ModelCheckpoint(monitor="val/mse_ft", mode="min", save_top_k=1)

    trainer = L.Trainer(
        max_epochs=250,
        logger=wandb_logger,
        accelerator="auto",
        callbacks=[checkpoint],
    )

    trainer.fit(model, data_module)

    data_module.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR))
