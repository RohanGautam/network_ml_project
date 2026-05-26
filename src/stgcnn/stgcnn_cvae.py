"""
Graph CVAE for NBA trajectory prediction — multimodal extension of the
deterministic Social-STGCNN pipeline in stgcnn_nba.py.

Motivation: the deterministic model is mean-seeking, so on genuinely multimodal
NBA futures it collapses to a blurred average (the persistent long-horizon
drift). The min-of-K oracle (~2.76) sits well below the deterministic mean
(~3.9), i.e. there is real headroom if the model can represent *multiple*
plausible futures. This is a SocialVAE-style conditional VAE — graph-based in
all three networks — bolted onto the components we already built:

  Prior      p(z | past)          ST-GCN encoder over the observed window
  Posterior  q(z | past, future)  ST-GCN encoder over the full window (train only)
  Decoder    p(future | z, past)  autoregressive graph rollout, conditioned on z

z is per-node (each agent has its own latent, coupled through the graph). Loss
is reconstruction MSE on the integrated absolute trajectory (matching the Kaggle
metric) + β·KL(q‖p) with linear warm-up. At inference we sample K futures from
the prior; for the single-shot submission we pick prior-mean vs sample-mean by
whichever wins on val, and log min-of-K as the oracle.

The dataset, sampler, datamodule, graph builder, st_gcn layers, metrics and
submission helper are all reused from stgcnn_nba.py.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import (
    compute_ade,
    compute_fde,
    compute_mse,
    compute_min_mse,
)
from stgcnn.stgcnn_nba import (
    st_gcn,
    _laplacian_from,
    rel_to_abs,
    NBADataModule,
    PROJECT_ROOT,
    TEST_DIR,
    SUBMISSION_DIR,
)

dotenv.load_dotenv(dotenv.find_dotenv())


# ── Graph modules ───────────────────────────────────────────────────────────


def _lap_seq(coords: Tensor) -> Tensor:
    """Batched per-frame position-space Laplacian: [B, T, N, 2] -> [B, T, N, N]."""
    B, T, N, _ = coords.shape
    return _laplacian_from(coords.reshape(B * T, N, 2)).reshape(B, T, N, N)


class GraphTrajEncoder(nn.Module):
    """ST-GCN stack over a (features, per-frame graphs) sequence.

    forward(feat [B,C,T,N], A [B,R,T,N,N]) -> per-node hidden at the last frame
    [B, N, hidden] (summarizes the whole input window for that node).
    """

    def __init__(self, input_feat, hidden_feat, n_stgcnn=2, kernel_size=3, n_relations=1):
        super().__init__()
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, hidden_feat, (kernel_size, n_relations)))
        for _ in range(1, n_stgcnn):
            self.st_gcns.append(st_gcn(hidden_feat, hidden_feat, (kernel_size, n_relations)))

    def forward(self, feat: Tensor, A: Tensor) -> Tensor:
        for gcn in self.st_gcns:
            feat, A = gcn(feat, A)  # [B, hidden, T, N]
        return feat[:, :, -1, :].permute(0, 2, 1)  # [B, N, hidden]


class CondGraphGRUDecoder(nn.Module):
    """Autoregressive graph rollout conditioned on a per-node latent z.

    Same live position-space graph rollout as the deterministic decoder, but the
    latent z is added to the initial hidden state and injected at every step, so
    the sampled future is coherent and z-dependent. Emits displacement means.
    """

    def __init__(self, hidden, latent, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.hidden = hidden
        self.disp_embed = nn.Linear(2, hidden)
        self.graph_lin = nn.Linear(hidden, hidden)
        self.z_to_h = nn.Linear(latent, hidden)
        self.z_to_in = nn.Linear(latent, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, h0, z, pos0, vel0):
        # h0 [B,N,H]  z [B,N,L]  pos0/vel0 [B,N,2]
        B, N, H = h0.shape
        h = h0 + self.z_to_h(z)
        z_in = self.z_to_in(z)
        pos, d_prev = pos0, vel0
        outs = []
        for _ in range(self.pred_len):
            A = _laplacian_from(pos.detach())  # [B, N, N]
            msg = torch.einsum("bnm,bmh->bnh", A, self.graph_lin(h))
            inp = msg + self.disp_embed(d_prev) + z_in
            h = self.gru(inp.reshape(B * N, H), h.reshape(B * N, H)).reshape(B, N, H)
            d = self.head(h)  # [B, N, 2] displacement
            outs.append(d)
            pos = pos + d
            d_prev = d
        return torch.stack(outs, dim=1)  # [B, pred, N, 2] displacements


# ── Lightning module ─────────────────────────────────────────────────────────


class NBACVAELightningModel(L.LightningModule):
    def __init__(
        self,
        context_size: int = 8,
        horizon_size: int = 12,
        n_stgcnn: int = 2,
        hidden_feat: int = 64,
        latent_dim: int = 16,
        kernel_size: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        beta: float = 1.0,
        kl_warmup_epochs: int = 50,
        n_samples: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()
        # input_feat = 4: [dx, dy, isplayer, team]; pos-space graph → n_relations=1.
        self.past_encoder = GraphTrajEncoder(4, hidden_feat, n_stgcnn, kernel_size, 1)
        self.full_encoder = GraphTrajEncoder(4, hidden_feat, n_stgcnn, kernel_size, 1)
        self.prior_head = nn.Linear(hidden_feat, 2 * latent_dim)
        self.post_head = nn.Linear(hidden_feat, 2 * latent_dim)
        self.decoder = CondGraphGRUDecoder(hidden_feat, latent_dim, horizon_size)
        self.predict_mode = "prior_mean"  # set from val; "prior_mean" | "sample_mean"

    # --- building blocks ---

    def _vel0(self, X):
        return X[:, :2, -1, :].permute(0, 2, 1)  # [B, N, 2] last observed velocity

    def _encode_past(self, X, A):
        h = self.past_encoder(X, A)  # [B, N, H]
        mu, logvar = self.prior_head(h).chunk(2, dim=-1)
        return h, mu, logvar

    def _encode_full(self, X, A, V_tr, abs_target):
        # Append the future to the observed window for the posterior.
        ident = X[:, 2:4, -1:, :]  # [B, 2, 1, N] static identity
        fut_disp = V_tr.permute(0, 3, 1, 2)  # [B, 2, pred, N]
        fut_ident = ident.expand(-1, -1, fut_disp.shape[2], -1)  # [B, 2, pred, N]
        fut_feat = torch.cat([fut_disp, fut_ident], dim=1)  # [B, 4, pred, N]
        full_feat = torch.cat([X, fut_feat], dim=2)  # [B, 4, obs+pred, N]
        fut_A = _lap_seq(abs_target).unsqueeze(1)  # [B, 1, pred, N, N]
        full_A = torch.cat([A, fut_A], dim=2)  # [B, 1, obs+pred, N, N]
        h = self.full_encoder(full_feat, full_A)
        mu, logvar = self.post_head(h).chunk(2, dim=-1)
        return mu, logvar

    def _decode_abs(self, h, z, last_pos, vel0):
        disp = self.decoder(h, z, last_pos, vel0)  # [B, pred, N, 2]
        return rel_to_abs(disp, last_pos, time_dim=1)  # [B, pred, N, 2]

    @staticmethod
    def _kl(mu_q, lv_q, mu_p, lv_p):
        # KL(N(mu_q,σ_q²) ‖ N(mu_p,σ_p²)) per latent dim; sum over L, mean over B,N.
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1)
        return kl.sum(-1).mean()

    # --- steps ---

    def training_step(self, batch, batch_idx):
        X, A, V_tr, last_pos, abs_target = batch
        h, mu_p, lv_p = self._encode_past(X, A)
        mu_q, lv_q = self._encode_full(X, A, V_tr, abs_target)
        z = mu_q + torch.randn_like(mu_q) * (0.5 * lv_q).exp()  # reparameterize
        abs_pred = self._decode_abs(h, z, last_pos, self._vel0(X))

        recon = ((abs_pred - abs_target) ** 2).mean()
        kl = self._kl(mu_q, lv_q, mu_p, lv_p)
        beta = self.hparams.beta * min(1.0, (self.current_epoch + 1) / self.hparams.kl_warmup_epochs)
        loss = recon + beta * kl

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/recon", recon, on_epoch=True, prog_bar=True)
        self.log("train/kl", kl, on_epoch=True)
        self.log("train/beta", beta, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, A, V_tr, last_pos, abs_target = batch
        B, T, N, _ = abs_target.shape
        h, mu_p, lv_p = self._encode_past(X, A)
        vel0 = self._vel0(X)

        target_flat = abs_target.permute(1, 0, 2, 3).reshape(T, B * N, 2)

        # (a) prior mean — deterministic single shot
        abs_mean = self._decode_abs(h, mu_p, last_pos, vel0)
        mean_flat = abs_mean.permute(1, 0, 2, 3).reshape(T, B * N, 2)

        # (b)+(c) K prior samples → sample-mean and min-of-K oracle
        k = self.hparams.n_samples
        samples = []
        for _ in range(k):
            z = mu_p + torch.randn_like(mu_p) * (0.5 * lv_p).exp()
            samples.append(self._decode_abs(h, z, last_pos, vel0))
        samp = torch.stack(samples)  # [K, B, pred, N, 2]
        sampmean_flat = samp.mean(0).permute(1, 0, 2, 3).reshape(T, B * N, 2)
        samp_flat = samp.permute(0, 2, 1, 3, 4).reshape(k, T, B * N, 2)

        self.log("val/mse_ft", compute_mse(mean_flat, target_flat), on_epoch=True, prog_bar=True)
        self.log("val/mse_ft_sampmean", compute_mse(sampmean_flat, target_flat), on_epoch=True, prog_bar=True)
        self.log(f"val/min_mse_ft_k{k}", compute_min_mse(samp_flat, target_flat), on_epoch=True, prog_bar=True)
        self.log("val/ade_ft", compute_ade(mean_flat, target_flat), on_epoch=True)
        self.log("val/fde_ft", compute_fde(mean_flat, target_flat), on_epoch=True)

        # Ball-vs-players on the prior mean (ball = isplayer==0).
        sq = ((abs_mean - abs_target) ** 2).mean(-1)  # [B, T, N]
        ball = (X[:, 2, 0, :] == 0).unsqueeze(1).expand(B, T, N)
        self.log("val/mse_ball", sq[ball].mean(), on_epoch=True)
        self.log("val/mse_players", sq[~ball].mean(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    # --- inference (reused by NBADataModule.get_kaggle_submission) ---

    @torch.no_grad()
    def _abs_pred(self, X, A, last_pos):
        """Absolute prediction on device per self.predict_mode → [B, pred, N, 2]."""
        h, mu_p, lv_p = self._encode_past(X, A)
        vel0 = self._vel0(X)
        if self.predict_mode == "sample_mean":
            k = self.hparams.n_samples
            samples = [
                self._decode_abs(h, mu_p + torch.randn_like(mu_p) * (0.5 * lv_p).exp(), last_pos, vel0)
                for _ in range(k)
            ]
            return torch.stack(samples).mean(0)
        return self._decode_abs(h, mu_p, last_pos, vel0)  # prior mean

    @torch.no_grad()
    def predict_abs_mean(self, X, A, last_pos):
        self.eval()
        last_pos = last_pos.to(self.device)
        return self._abs_pred(X.to(self.device), A.to(self.device), last_pos).cpu()

    @torch.no_grad()
    def tta_abs_pred(self, X, A, last_pos):
        """Batched 4-way court-symmetry TTA → [B, pred, N, 2] (on device)."""
        self.eval()
        preds = []
        for axes in [(), (0,), (1,), (0, 1)]:
            Xf, lp = X.clone(), last_pos.clone()
            for ax in axes:
                Xf[:, ax] = -Xf[:, ax]
                lp[:, :, ax] = -lp[:, :, ax]
            ap = self._abs_pred(Xf, A, lp)
            for ax in axes:
                ap[..., ax] = -ap[..., ax]
            preds.append(ap)
        return torch.stack(preds).mean(0)


# ── Eval + submission ──────────────────────────────────────────────────────────


def eval_and_submit(model, data_module, logger=None):
    """Pick prior-mean vs sample-mean (and TTA) by val mse_ft, then submit the best."""
    model.eval()
    dev = model.device
    # accumulate squared error for each strategy
    keys = ["prior_mean", "sample_mean"]
    se = {k: 0.0 for k in keys}
    se_tta = {k: 0.0 for k in keys}
    n = 0.0
    for X, A, V_tr, last_pos, abs_target in data_module.val_dataloader():
        X, A, last_pos, abs_target = (t.to(dev) for t in (X, A, last_pos, abs_target))
        for mode in keys:
            model.predict_mode = mode
            plain = model._abs_pred(X, A, last_pos)
            tta = model.tta_abs_pred(X, A, last_pos)
            se[mode] += ((plain - abs_target) ** 2).sum().item()
            se_tta[mode] += ((tta - abs_target) ** 2).sum().item()
        n += abs_target.numel()

    results = {}
    for mode in keys:
        results[(mode, False)] = se[mode] / n
        results[(mode, True)] = se_tta[mode] / n
    (best_mode, best_tta), best_val = min(results.items(), key=lambda kv: kv[1])
    print("[val] mse_ft by strategy:")
    for (mode, tta), v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"    {mode:12s} tta={tta!s:5s} -> {v:.4f}")
    print(f"  best: {best_mode} tta={best_tta} ({best_val:.4f})")
    if logger is not None:
        logger.experiment.summary["val/mse_ft_best"] = best_val

    model.predict_mode = best_mode
    data_module.get_kaggle_submission(model, str(TEST_DIR), str(SUBMISSION_DIR), tta=best_tta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Eval-only from checkpoint.")
    args = parser.parse_args()

    L.seed_everything(0)

    data_module = NBADataModule(
        split_path=str(PROJECT_ROOT / "splits" / "fold0.json"),
        batch_size=128,
        graph_space="pos",  # CVAE uses position-space graphs (n_relations=1)
    )

    if args.ckpt:
        model = NBACVAELightningModel.load_from_checkpoint(args.ckpt)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        data_module.setup()
        eval_and_submit(model, data_module)
        sys.exit(0)

    model = NBACVAELightningModel()

    h = model.hparams
    wandb_logger = WandbLogger(
        project="NML_base",
        name=f"stgcnn_cvae_h{h.hidden_feat}_z{h.latent_dim}_b{h.beta}",
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(PROJECT_ROOT / "checkpoints" / "cvae"),
        filename="best", monitor="val/mse_ft", mode="min", save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=250,
        logger=wandb_logger,
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=[checkpoint],
    )
    trainer.fit(model, data_module)

    eval_and_submit(model, data_module, logger=wandb_logger)
