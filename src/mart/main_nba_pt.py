"""Train MART on NBA .pt data from our split manifest.

Follows the same training setup as MART/main_nba.py but reads .pt sequences,
z-scores per split, uses past=8/future=12, and logs to wandb.

Example:
    python main_nba_pt.py \\
        --config configs/mart_nba_pt.yaml \\
        --split_path ../network_ml_project/splits/fold0.json \\
        --model_name mart_pt_run1 \\
        --gpu 0

Checkpoint saved to ./checkpoints/<model_name>.ckpt — contains state_dict,
full config, mu and sigma for submit_nba_pt.py.
"""

import argparse
import math
import os
import random
import sys

import dotenv
import numpy as np
import torch
import wandb

dotenv.load_dotenv(dotenv.find_dotenv())

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from utils import load_config, setup_seed, get_th
from models.mart import MART
from models.mart_id import MART_ID
from loaders.dataloader_nba_pt import (
    MARTNBAPTDataset,
    WindowSampler,
    WindowEvalSampler,
    compute_xy_stats,
    load_split_files,
)
from loaders.dataloader_nba_pt_hoops import (
    MARTNBAPTDataset as MARTNBAPTDatasetHoops,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID as HOOPS_ID,
)


if not torch.cuda.is_available():
    # prt.py / hrt.py have hardcoded .cuda() calls on internal tensors,
    # so we patch it out when running on CPU
    print("CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat")
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def parse_args():
    p = argparse.ArgumentParser(description="MART on network_ml_project NBA data")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", type=str, default="configs/mart_nba_pt.yaml")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name for the saved checkpoint at ./checkpoints/<model_name>.ckpt",
    )
    p.add_argument(
        "--split_path",
        type=str,
        required=False,
        default="../network_ml_project/splits/fold0.json",
        help="Path to network_ml_project splits/<name>.json",
    )
    p.add_argument("--num_workers", type=int, default=4)
    # min_ade is MART's default; mean_mse is closer to Kaggle's scoring
    p.add_argument(
        "--loss",
        type=str,
        default="min_ade",
        choices=["min_ade", "mean_ade", "min_mse", "mean_mse"],
        help="Training loss: {min|mean} over K of {ADE|MSE}.",
    )
    p.add_argument(
        "--use_hoops",
        action="store_true",
        help="Append 2 static basket-hoop nodes as extra agents.",
    )
    p.add_argument(
        "--aug_court_mirror",
        action="store_true",
        help="Random court-symmetry reflection per batch sample at train time.",
    )
    # iso_norm is required for rotation augmentation to be a valid isometry
    p.add_argument(
        "--iso_norm",
        action="store_true",
        help="Isotropic (shared scalar) z-score std; needed for --aug_rotate.",
    )
    p.add_argument(
        "--aug_rot_deg",
        type=float,
        default=0.0,
        help="Max rotation magnitude in degrees (0 = off; 180 = full circle).",
    )
    p.add_argument(
        "--aug_jitter",
        type=float,
        default=0.0,
        help="Std of Gaussian jitter added to past positions (normed units).",
    )
    p.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override config num_epochs (e.g. long training).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override config lr (e.g. lower LR for fine-tuning).",
    )
    p.add_argument(
        "--dropout", type=float, default=None, help="Override config dropout."
    )
    p.add_argument(
        "--scheduler_type",
        type=str,
        default=None,
        help="Override config scheduler_type (e.g. CosineAnnealingWarmRestarts).",
    )
    p.add_argument(
        "--sgdr_t0",
        type=int,
        default=None,
        help="SGDR first-cycle length in epochs (CosineAnnealingWarmRestarts).",
    )
    p.add_argument(
        "--sgdr_tmult",
        type=int,
        default=None,
        help="SGDR cycle-length multiplier per restart (default 1).",
    )
    p.add_argument(
        "--hoop_feats",
        action="store_true",
        help="Inject hoop-relative offset vectors as extra input features. "
        "Requires --use_hoops (hoop positions are taken from the 2 "
        "landmark nodes already in x_abs, so augmentation is automatic).",
    )
    p.add_argument(
        "--ball_dist",
        action="store_true",
        help="Inject scalar Euclidean distance to the ball as an extra input "
        "feature for every agent at every past timestep. Ball is always "
        "at canonical index 10. Adds 1 dim to extra_input_dim.",
    )
    p.add_argument(
        "--edge_type_emb",
        action="store_true",
        help="Add a learned edge-type embedding bias to the RT pair encoder's "
        "initial edge features. Types encode basketball structure: "
        "same-team, opponent, player↔ball, agent↔hoop, self.",
    )
    p.add_argument(
        "--curriculum",
        action="store_true",
        help="Curriculum loss: use --loss (default min_ade) for the first "
        "--curriculum_switch epochs, then switch to mean_mse.",
    )
    p.add_argument(
        "--curriculum_switch",
        type=int,
        default=None,
        help="Epoch at which to switch from --loss to mean_mse. "
        "Defaults to half of num_epochs if not set.",
    )
    p.add_argument(
        "--curriculum_lr_reset",
        action="store_true",
        help="At the curriculum switch epoch, rebuild the LR scheduler "
        "with T_max = remaining epochs so each phase gets its own "
        "full cosine decay instead of sharing one stretched curve.",
    )
    p.add_argument(
        "--curriculum_lr2",
        type=float,
        default=None,
        help="LR to use for phase 2 when --curriculum_lr_reset is set. "
        "Defaults to --lr (same as phase 1) if not specified.",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint whose state_dict is loaded into the "
        "model before training starts. Optimizer and scheduler are "
        "freshly initialised from the current CLI args, so LR and "
        "loss can differ from the source run.",
    )
    p.add_argument(
        "--soft_wta_temp",
        type=float,
        default=0.5,
        help="Temperature for soft-WTA loss (--loss soft_wta). "
        "Lower = closer to min-of-K; higher = closer to mean-of-K.",
    )
    p.add_argument(
        "--laplace_nll",
        action="store_true",
        help="Use Laplace NLL loss. Changes decoder output to 4D "
        "(loc + scale). Best mode selected by ADE on loc.",
    )
    p.add_argument(
        "--cfi",
        action="store_true",
        help="Enable Cross-modal Future Interaction decoder. Adds Branch 2 "
        "with self-attention over K×N mode-agent tokens. "
        "Loss = loss(branch1) + loss(branch2). Val uses branch2 mean-of-K.",
    )
    p.add_argument("--wandb_project", type=str, default="NML_base")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--wandb_mode", type=str, default="online", help="online | offline | disabled"
    )
    return p.parse_args()


def _x_rel_from_x_abs(x_abs):
    """Velocity in MART's convention: x_rel[t] = x_abs[t] - x_abs[t-1], x_rel[0] = x_rel[1]."""
    x_rel = torch.zeros_like(x_abs)
    x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    x_rel[:, :, 0] = x_rel[:, :, 1]
    return x_rel


def _model_forward(
    model, x_abs, x_rel, agent_ids, extra_feats=None, mu=None, sigma=None
):
    """Dispatch to MART_ID (takes agent_ids) vs stock MART (ignores them)."""
    if isinstance(model, MART_ID):
        return model(
            x_abs, x_rel, agent_ids, extra_feats=extra_feats, mu=mu, sigma=sigma
        )
    return model(x_abs, x_rel, extra_feats=extra_feats, mu=mu, sigma=sigma)


def _compute_ball_dist_feats(x_abs):
    """Euclidean distance from every agent to the ball at every past timestep.

    Ball is always at canonical index 10 ([TeamA(5), TeamB(5), Ball]).
    Returns [B, N, T_p, 1].
    """
    ball_pos = x_abs[:, 10:11, :, :]
    dist = torch.norm(x_abs - ball_pos, dim=-1, keepdim=True)
    return dist


def _compute_hoop_feats(x_abs, n_real):
    """Hoop-relative offset vectors for every agent at every past timestep.

    Hoop nodes sit at indices n_real and n_real+1 in x_abs. Since we derive
    offsets from those positions directly, they stay consistent under any
    augmentation that was applied to x_abs.

    Returns [B, N, T_p, 4] = [off_hoop1_x, off_hoop1_y, off_hoop2_x, off_hoop2_y].
    """
    hoop1 = x_abs[:, n_real : n_real + 1, :, :]
    hoop2 = x_abs[:, n_real + 1 : n_real + 2, :, :]
    off1 = hoop1 - x_abs
    off2 = hoop2 - x_abs
    return torch.cat([off1, off2], dim=-1)


def _augment_court(x_abs, y, center, rot_max_rad=0.0, mirror=False, jitter_std=0.0):
    """Random O(2) court-symmetry augmentation applied per batch instance (training only).

    Composes a per-sample rotation and/or independent x/y reflection about the
    court center, so pairwise distances and agent identities are preserved.
    The same transform is applied to both past and future so the window stays
    coherent; x_rel is recomputed from the transformed x_abs in train_one_epoch.

    Rotation is only valid under isotropic normalization (--iso_norm): without it,
    a rotation in raw feet becomes a shear in the normed frame.

    Court center in normed space: center = -mu / sigma (raw origin is court center).

    Args:
        x_abs:       [B, N, T_p, 2]  z-scored past positions
        y:           [B, N, T_f, 2]  z-scored future positions
        center:      [2]             normed court center on device
        rot_max_rad: angle drawn from U(-rot_max_rad, rot_max_rad). 0 = off.
        mirror:      if True, independent 50/50 x/y reflections.
        jitter_std:  Gaussian noise std added to past only (input regularizer). 0 = off.
    Returns:
        Transformed (x_abs, y).
    """
    B = x_abs.shape[0]
    device = x_abs.device

    if mirror:
        sx = torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
        sy = torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
    else:
        sx = torch.ones(B, device=device)
        sy = torch.ones(B, device=device)

    if rot_max_rad > 0.0:
        theta = (torch.rand(B, device=device) * 2.0 - 1.0) * rot_max_rad
    else:
        theta = torch.zeros(B, device=device)
    cos, sin = torch.cos(theta), torch.sin(theta)

    # 2x2 per-sample map M = R(theta) @ diag(sx, sy), applied about center
    m00 = (cos * sx).view(B, 1, 1)
    m01 = (-sin * sy).view(B, 1, 1)
    m10 = (sin * sx).view(B, 1, 1)
    m11 = (cos * sy).view(B, 1, 1)
    c = center.view(1, 1, 1, 2)

    def _apply(z):
        zc = z - c
        xc, yc = zc[..., 0], zc[..., 1]
        nx = m00 * xc + m01 * yc
        ny = m10 * xc + m11 * yc
        return torch.stack([nx, ny], dim=-1) + c

    x_abs = _apply(x_abs)
    y = _apply(y)

    if jitter_std > 0.0:
        x_abs = x_abs + torch.randn_like(x_abs) * jitter_std

    return x_abs, y


def compute_loss(y_pred, y_exp, loss_type, soft_wta_temp=0.5):
    """Aggregate K hypotheses into a scalar loss.

    y_pred: [B, N, K, T_f, 2] (or 4D for laplace_nll)
    y_exp:  [B, N, 1, T_f, 2]

    loss_type options:
        min_ade / mean_ade  — L2 distance, min or mean over K
        min_mse / mean_mse  — squared error, min or mean over K
        soft_wta            — temperature-weighted softmin; interpolates min and mean
        laplace_nll         — Laplace NLL on best mode; y_pred must be 4D (loc + scale)
    """
    if loss_type == "soft_wta":
        per_k = torch.norm(y_pred - y_exp, dim=-1).mean(dim=3)
        weights = torch.nn.functional.softmin(per_k / soft_wta_temp, dim=2).detach()
        return (weights * per_k).sum(dim=2).mean()

    if loss_type == "laplace_nll":
        MIN_SCALE = 1e-3
        loc = y_pred[..., :2]
        scale = torch.nn.functional.softplus(y_pred[..., 2:]) + MIN_SCALE
        per_k = torch.norm(loc - y_exp, dim=-1).mean(dim=3)
        best_k = per_k.argmin(dim=2)
        B, N, K, T_f, _ = loc.shape
        idx = best_k.view(B, N, 1, 1, 1).expand(B, N, 1, T_f, 2)
        best_loc = loc.gather(2, idx).squeeze(2)
        best_scale = scale.gather(2, idx).squeeze(2)
        y_gt = y_exp.squeeze(2)
        nll = torch.log(2 * best_scale) + torch.abs(y_gt - best_loc) / best_scale
        return nll.mean()

    if loss_type in ("min_ade", "mean_ade"):
        per_k = torch.norm(y_pred - y_exp, dim=-1).mean(dim=3)
    elif loss_type in ("min_mse", "mean_mse"):
        per_k = ((y_pred - y_exp) ** 2).mean(dim=(3, 4))
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    if loss_type.startswith("min_"):
        reduced = per_k.min(dim=2)[0]
    else:
        reduced = per_k.mean(dim=2)

    return reduced.mean()


def train_one_epoch(
    epoch, model, optimizer, loader, opts, device, aug=None, mu=None, sigma=None
):
    """One training epoch.

    aug: dict with {center, rot_max_rad, mirror, jitter_std} for geometric
         augmentation, or None to skip.
    """
    model.train()
    total_n = 0
    total_loss = 0.0

    n_iter = len(loader)
    for i, (x_abs, y, agent_ids) in enumerate(loader):
        x_abs = x_abs.to(device)
        y = y.to(device)
        agent_ids = agent_ids.to(device)
        B, N, _, _ = x_abs.shape

        if aug is not None:
            x_abs, y = _augment_court(
                x_abs,
                y,
                aug["center"],
                rot_max_rad=aug["rot_max_rad"],
                mirror=aug["mirror"],
                jitter_std=aug["jitter_std"],
            )

        extra_parts = []
        if opts.get("hoop_feats", False):
            extra_parts.append(_compute_hoop_feats(x_abs, HOOPS_N_REAL_AGENTS))
        if opts.get("ball_dist", False):
            extra_parts.append(_compute_ball_dist_feats(x_abs))
        extra_feats = torch.cat(extra_parts, dim=-1) if extra_parts else None

        x_rel = _x_rel_from_x_abs(x_abs)
        fwd = _model_forward(model, x_abs, x_rel, agent_ids, extra_feats, mu, sigma)

        # CFI returns (branch1, branch2); use branch2 as the primary prediction
        cfi_mode = isinstance(fwd, tuple)
        y_pred = fwd[1] if cfi_mode else fwd

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

        y_exp = y[:, :, None, :, :]

        if opts.get("use_hoops", False):
            y_pred_loss = y_pred[:, :HOOPS_N_REAL_AGENTS]
            y_exp_loss = y_exp[:, :HOOPS_N_REAL_AGENTS]
        else:
            y_pred_loss = y_pred
            y_exp_loss = y_exp

        # switch to mean_mse after curriculum_switch
        effective_loss = opts.loss
        if opts.get("curriculum", False) and epoch >= opts.get(
            "curriculum_switch", opts.num_epochs // 2
        ):
            effective_loss = "mean_mse"

        loss = compute_loss(
            y_pred_loss,
            y_exp_loss,
            effective_loss,
            soft_wta_temp=opts.get("soft_wta_temp", 0.5),
        )

        # add branch1 loss so diversity is maintained before CFI refinement
        if cfi_mode:
            loc1 = fwd[0]
            loc1_loss = (
                loc1[:, :HOOPS_N_REAL_AGENTS] if opts.get("use_hoops", False) else loc1
            )
            loss = loss + compute_loss(
                loc1_loss,
                y_exp_loss,
                effective_loss,
                soft_wta_temp=opts.get("soft_wta_temp", 0.5),
            )

        optimizer.zero_grad()
        loss.backward()
        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
        optimizer.step()

        n_scored = HOOPS_N_REAL_AGENTS if opts.get("use_hoops", False) else N
        total_loss += loss.item() * B * n_scored
        total_n += B * n_scored

        if i % 100 == 0:
            th = get_th(opts, model)
            print(
                f"[TRAIN] Epoch {epoch + 1:03d}/{opts.num_epochs:03d} | "
                f"It {i + 1:04d}/{n_iter:04d} | "
                f"Loss {loss.item():.4f} | th {th} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_minADE_minFDE(model, loader, opts, device, split_name, mu=None, sigma=None):
    """Compute val metrics.

    Reports:
      val/mse_ft  — mean squared error in feet² on the mean-of-K prediction.
                    Matches Kaggle's scoring and EqMotion's val/mse_ft directly.
      minADE/minFDE — min-of-K in z-score space, kept for parity with MART paper.
    """
    model.eval()
    sum_ade = 0.0
    sum_fde = 0.0
    sum_loss = 0.0
    n_total = 0
    sse_ft = 0.0
    cnt_ft = 0

    use_ft = mu is not None and sigma is not None
    if use_ft:
        mu_b = mu.to(device).view(1, 1, 1, 2).float()
        sigma_b = sigma.to(device).view(1, 1, 1, 2).float()

    for x_abs, y, agent_ids in loader:
        x_abs = x_abs.to(device)
        y = y.to(device)
        agent_ids = agent_ids.to(device)
        B, N, _, _ = x_abs.shape

        extra_parts = []
        if opts.get("hoop_feats", False):
            extra_parts.append(_compute_hoop_feats(x_abs, HOOPS_N_REAL_AGENTS))
        if opts.get("ball_dist", False):
            extra_parts.append(_compute_ball_dist_feats(x_abs))
        extra_feats = torch.cat(extra_parts, dim=-1) if extra_parts else None

        x_rel = _x_rel_from_x_abs(x_abs)
        fwd = _model_forward(model, x_abs, x_rel, agent_ids, extra_feats, mu, sigma)
        y_pred = fwd[1] if isinstance(fwd, tuple) else fwd

        # strip scale channels for Laplace NLL — only loc matters for metrics
        if y_pred.shape[-1] == 4:
            y_pred = y_pred[..., :2]

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

        y_exp = y[:, :, None, :, :]

        # exclude hoop nodes from val metrics — they're static and inflate scores
        if opts.get("use_hoops", False):
            y_pred = y_pred[:, :HOOPS_N_REAL_AGENTS]
            y_exp = y_exp[:, :HOOPS_N_REAL_AGENTS]
            y_real = y[:, :HOOPS_N_REAL_AGENTS]
            N_scored = HOOPS_N_REAL_AGENTS
        else:
            y_real = y
            N_scored = N

        per_step_err = torch.norm(y_pred - y_exp, dim=-1)
        ade_perK = per_step_err.mean(dim=3)
        min_ade = ade_perK.min(dim=2)[0]
        fde_perK = per_step_err[:, :, :, -1]
        min_fde = fde_perK.min(dim=2)[0]
        loss_val = min_ade.mean()
        sum_ade += min_ade.sum().item()
        sum_fde += min_fde.sum().item()
        sum_loss += loss_val.item() * B * N_scored
        n_total += B * N_scored

        if use_ft:
            y_mean = y_pred.mean(dim=2)
            y_mean_ft = y_mean * sigma_b + mu_b
            y_tgt_ft = y_real * sigma_b + mu_b
            sse_ft += ((y_mean_ft - y_tgt_ft) ** 2).sum().item()
            cnt_ft += y_mean_ft.numel()

    avg_ade = sum_ade / max(n_total, 1)
    avg_fde = sum_fde / max(n_total, 1)
    avg_loss = sum_loss / max(n_total, 1)
    mse_ft = sse_ft / cnt_ft if use_ft and cnt_ft > 0 else float("nan")

    if use_ft:
        print(
            f"[{split_name.upper()}] mse_ft = {mse_ft:.4f} ft²  "
            f"| minADE/minFDE (z-score) @ {opts.future_length} steps: "
            f"{avg_ade:.4f} / {avg_fde:.4f}"
        )
    else:
        print(
            f"[{split_name.upper()}] minADE/minFDE @ {opts.future_length} steps: "
            f"{avg_ade:.4f} / {avg_fde:.4f}"
        )
    return {"loss": avg_loss, "minADE": avg_ade, "minFDE": avg_fde, "mse_ft": mse_ft}


def main():
    args = parse_args()
    setup_seed(args.seed)
    print("[INFO] seed:", args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    opts = load_config(args.config)
    opts.loss = args.loss
    opts.use_hoops = args.use_hoops
    opts.hoop_feats = args.hoop_feats
    opts.ball_dist = args.ball_dist
    opts.edge_type_emb = args.edge_type_emb
    opts.cfi = args.cfi
    opts.curriculum = args.curriculum
    opts.curriculum_switch = args.curriculum_switch
    opts.curriculum_lr_reset = args.curriculum_lr_reset
    opts.curriculum_lr2 = args.curriculum_lr2
    opts.soft_wta_temp = args.soft_wta_temp
    opts.laplace_nll = args.laplace_nll
    if args.laplace_nll:
        opts.loss = "laplace_nll"
    opts.extra_input_dim = (4 if args.hoop_feats else 0) + (1 if args.ball_dist else 0)
    opts.aug_court_mirror = args.aug_court_mirror
    opts.iso_norm = args.iso_norm
    opts.aug_rot_deg = args.aug_rot_deg
    opts.aug_jitter = args.aug_jitter
    if args.num_epochs is not None:
        opts.num_epochs = args.num_epochs
    if args.lr is not None:
        opts.lr = args.lr
    if args.dropout is not None:
        opts.dropout = args.dropout
    if args.scheduler_type is not None:
        opts.scheduler_type = args.scheduler_type
    if args.sgdr_t0 is not None:
        opts.sgdr_t0 = args.sgdr_t0
    if args.sgdr_tmult is not None:
        opts.sgdr_tmult = args.sgdr_tmult

    # rotation in z-score space is only valid when x and y use the same std
    if opts.aug_rot_deg > 0.0 and not opts.iso_norm:
        print("[WARN] --aug_rot_deg > 0 requires isotropic norm; enabling --iso_norm.")
        opts.iso_norm = True

    if opts.hoop_feats and not opts.use_hoops:
        raise ValueError(
            "--hoop_feats requires --use_hoops (hoop positions come from the landmark nodes in x_abs)."
        )

    if opts.curriculum and opts.curriculum_switch is None:
        opts.curriculum_switch = opts.num_epochs // 2
    if opts.curriculum:
        print(
            f"[INFO] curriculum: {opts.loss} for epochs 0–{opts.curriculum_switch - 1}, "
            f"then mean_mse for epochs {opts.curriculum_switch}–{opts.num_epochs - 1}"
        )
        if opts.curriculum_lr_reset:
            print(
                f"[INFO] curriculum_lr_reset: scheduler rebuilt at epoch "
                f"{opts.curriculum_switch} with T_max={opts.num_epochs - opts.curriculum_switch}"
            )

    print(f"[INFO] training loss: {opts.loss}")
    print(f"[INFO] use_hoops: {opts.use_hoops}")
    print(
        f"[INFO] hoop_feats: {opts.hoop_feats}  ball_dist: {opts.ball_dist}  "
        f"extra_input_dim={opts.extra_input_dim}"
    )
    print(f"[INFO] iso_norm: {opts.iso_norm}")
    print(
        f"[INFO] aug: mirror={opts.aug_court_mirror} rot_deg={opts.aug_rot_deg} "
        f"jitter={opts.aug_jitter}"
    )
    print("[INFO] opts:", opts)

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.model_name}.ckpt")
    print(f"[INFO] final-epoch checkpoint will be saved to: {ckpt_path}")

    # in test mode, trust use_hoops from the checkpoint rather than CLI
    if args.test and os.path.isfile(ckpt_path):
        ckpt_opts = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        ).get("opts", {})
        ckpt_uh = bool(ckpt_opts.get("use_hoops", False))
        if ckpt_uh != opts.use_hoops:
            print(
                f"[INFO] --test: overriding --use_hoops={opts.use_hoops} "
                f"with checkpoint value {ckpt_uh}"
            )
            opts.use_hoops = ckpt_uh

    # data
    train_files, val_files = load_split_files(args.split_path)
    print(f"[INFO] split: {len(train_files)} train, {len(val_files)} val")

    mu, sigma = compute_xy_stats(train_files, iso=opts.iso_norm)
    print(
        f"[INFO] norm stats (iso={opts.iso_norm}): mu={mu.tolist()}, sigma={sigma.tolist()}"
    )

    # court center in normed space is -mu/sigma (raw origin = court center)
    if opts.aug_court_mirror or opts.aug_rot_deg > 0.0 or opts.aug_jitter > 0.0:
        aug = {
            "center": (-mu / sigma).to(device),
            "rot_max_rad": math.radians(opts.aug_rot_deg),
            "mirror": bool(opts.aug_court_mirror),
            "jitter_std": float(opts.aug_jitter),
        }
        print(
            f"[INFO] augmentation enabled: center={aug['center'].tolist()} "
            f"rot_max_rad={aug['rot_max_rad']:.3f} mirror={aug['mirror']} "
            f"jitter={aug['jitter_std']}"
        )
    else:
        aug = None

    DatasetCls = MARTNBAPTDatasetHoops if opts.use_hoops else MARTNBAPTDataset
    train_set = DatasetCls(
        train_files,
        mu,
        sigma,
        opts.past_length,
        opts.future_length,
    )
    val_set = DatasetCls(
        val_files,
        mu,
        sigma,
        opts.past_length,
        opts.future_length,
    )
    print(f"[INFO] usable sequences: train={len(train_set)}, val={len(val_set)}")

    train_sampler = WindowSampler(
        opts.batch_size,
        train_set.max_start,
        seed=args.seed,
        shuffle=True,
    )
    # deterministic 8-window-per-sequence val sampler, matches EqMotion's full-val
    val_sampler = WindowEvalSampler(val_set.max_start, windows_per_seq=8)
    print(f"[INFO] val windows: {len(val_sampler)} (deterministic 8/seq)")

    train_loader = DataLoader(
        train_set,
        batch_size=opts.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=opts.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # model
    use_id = bool(opts.get("use_entity_embed", False))
    if use_id:
        # hoops add a 4th entity class (id=HOOPS_ID), so bump num_entity_types
        n_entity_types = (HOOPS_ID + 1) if opts.use_hoops else 3
        print(
            f"[INFO] using MART_ID with embed_dim={opts.embed_dim}, "
            f"num_entity_types={n_entity_types}"
        )
        model = MART_ID(opts, num_entity_types=n_entity_types).to(device)
    else:
        print("[INFO] using stock MART (no entity embedding)")
        model = MART(opts).to(device)
    print(model)
    print(f"[INFO] params: {sum(p.numel() for p in model.parameters())}")

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[INFO] resumed weights from: {args.resume_from}")

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)
    if opts.scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=opts.decay_step,
            gamma=opts.decay_gamma,
        )
    elif opts.scheduler_type == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opts.milestones,
            gamma=opts.decay_gamma,
        )
    elif opts.scheduler_type == "CosineAnnealingLR":
        # eta_min = lr*0.02 matches EqMotion's schedule for fair comparison
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opts.num_epochs,
            eta_min=opts.lr * 0.02,
        )
    elif opts.scheduler_type == "CosineAnnealingWarmRestarts":
        # SGDR warm restarts; cycle lengths in epoch units, eta_min matches above
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=opts.sgdr_t0,
            T_mult=opts.get("sgdr_tmult", 1),
            eta_min=opts.lr * 0.02,
        )
    else:
        scheduler = None

    # test mode
    if args.test:
        print(f"[INFO] Loading model from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        eval_minADE_minFDE(
            model,
            val_loader,
            opts,
            device,
            split_name="val",
            mu=mu,
            sigma=sigma,
        )
        return

    # training loop
    val_metrics = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{args.model_name}_best.ckpt")
    best_mse = float("inf")
    # for SGDR, snapshot at the end of each cosine cycle (detected by LR jumping up)
    is_sgdr = opts.scheduler_type == "CosineAnnealingWarmRestarts"
    snapshot_dir = os.path.join(ckpt_dir, f"{args.model_name}_snapshots")
    if is_sgdr:
        os.makedirs(snapshot_dir, exist_ok=True)
    n_snapshots = 0

    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={**dict(opts), "split_path": args.split_path, "seed": args.seed},
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    ):
        for epoch in range(opts.num_epochs):
            train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                epoch,
                model,
                optimizer,
                train_loader,
                opts,
                device,
                aug,
                mu=mu,
                sigma=sigma,
            )
            val_metrics = eval_minADE_minFDE(
                model,
                val_loader,
                opts,
                device,
                split_name="val",
                mu=mu,
                sigma=sigma,
            )

            # rebuild cosine scheduler at switch so phase 2 gets a fresh full decay
            if (
                opts.get("curriculum_lr_reset", False)
                and opts.get("curriculum", False)
                and epoch == opts.get("curriculum_switch", -1)
                and opts.get("scheduler_type") == "CosineAnnealingLR"
            ):
                remaining = opts.num_epochs - epoch
                phase2_lr = opts.get("curriculum_lr2") or opts.lr
                for pg in optimizer.param_groups:
                    pg["lr"] = phase2_lr
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=remaining,
                    eta_min=phase2_lr * 0.02,
                )
                print(
                    f"[LR-RESET] epoch {epoch}: rebuilt CosineAnnealingLR "
                    f"T_max={remaining}, lr reset to {phase2_lr:.2e}"
                )

            lr_before = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            lr_after = optimizer.param_groups[0]["lr"]

            # LR increase after step() means a cosine cycle just ended; snapshot the model
            # (1.1x threshold avoids false positives from float noise at the cosine bottom)
            if is_sgdr and lr_after > lr_before * 1.1:
                snap_path = os.path.join(snapshot_dir, f"cycle_{n_snapshots:02d}.ckpt")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "opts": dict(opts),
                        "mu": mu,
                        "sigma": sigma,
                        "val_mse_ft": val_metrics["mse_ft"],
                    },
                    snap_path,
                )
                print(
                    f"[SGDR] cycle {n_snapshots} ended at epoch {epoch} "
                    f"(val/mse_ft {val_metrics['mse_ft']:.4f}) -> {snap_path}"
                )
                n_snapshots += 1

            if val_metrics["mse_ft"] < best_mse:
                best_mse = val_metrics["mse_ft"]
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "opts": dict(opts),
                        "mu": mu,
                        "sigma": sigma,
                        "val_mse_ft": val_metrics["mse_ft"],
                        "val_minADE": val_metrics["minADE"],
                        "val_minFDE": val_metrics["minFDE"],
                    },
                    best_ckpt_path,
                )

            log = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "th": get_th(opts, model),
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/mse_ft": val_metrics["mse_ft"],
                "val/minADE": val_metrics["minADE"],
                "val/minFDE": val_metrics["minFDE"],
            }
            wandb.log(log)

            print(
                f"[INFO] Epoch {epoch:03d} done | train_loss {train_loss:.4f} | "
                f"val/mse_ft {val_metrics['mse_ft']:.4f} ft² | "
                f"val_minADE {val_metrics['minADE']:.4f}"
            )

        # save final checkpoint
        torch.save(
            {
                "epoch": opts.num_epochs - 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "opts": dict(opts),
                "mu": mu,
                "sigma": sigma,
                "val_minADE": val_metrics["minADE"],
                "val_minFDE": val_metrics["minFDE"],
                "val_mse_ft": val_metrics["mse_ft"],
            },
            ckpt_path,
        )
        print(
            f"\n[INFO] Training complete. Saved final-epoch checkpoint to {ckpt_path}"
        )
        print(
            f"[INFO] Best-by-val checkpoint: {best_ckpt_path} (val/mse_ft {best_mse:.4f})"
        )
        print("=== Final val metrics (last epoch) ===")
        print(f"  val/mse_ft : {val_metrics['mse_ft']:.4f} ft²  (Kaggle-comparable)")
        print(f"  val/minADE : {val_metrics['minADE']:.4f}  (normalized, min-of-K)")
        print(f"  val/minFDE : {val_metrics['minFDE']:.4f}  (normalized, min-of-K)")
        print(f"  val/loss   : {val_metrics['loss']:.4f}")
        print(
            "[INFO] For metrics in feet (denormalized, ADE/FDE/MSE + min-of-K "
            "oracle), run eval.py against the saved checkpoint."
        )


if __name__ == "__main__":
    main()
