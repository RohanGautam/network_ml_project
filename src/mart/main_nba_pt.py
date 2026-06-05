"""Train MART from scratch on network_ml_project's .pt NBA data.

Mirrors MART/main_nba.py (same model, same min-of-K loss, same scheduler),
but:
    - reads .pt sequences via the network_ml_project split manifest
    - z-scores (x, y) using stats computed from the train split
    - uses past=8, future=12 (Kaggle horizon)
    - canonical agent order [TeamA(5), TeamB(5), Ball(1)]
    - wandb logging (parallel to GroupNet/groupnet/train_hyper_nba_pt.py)
    - last-epoch checkpoint (no best-on-val tracking)

Example:
    python main_nba_pt.py \\
        --config configs/mart_nba_pt.yaml \\
        --split_path ../network_ml_project/splits/fold0.json \\
        --model_name mart_pt_run1 \\
        --gpu 0

The final-epoch checkpoint is saved to
    ./checkpoints/<model_name>.ckpt
and contains everything needed to re-instantiate the model and run
submit_nba_pt.py (state_dict, full config, mu, sigma).
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

# WANDB_API_KEY is loaded from the project's .env (same pattern as the other
# training scripts in this repo). To override, set the env var before launch
# or run `wandb login`.
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
# Hoops variant: 13-agent dataset with 2 static basket nodes appended after the
# ball. Imported eagerly (cheap) but only used when --use_hoops is set; the
# default 11-agent pipeline above is untouched.
from loaders.dataloader_nba_pt_hoops import (
    MARTNBAPTDataset as MARTNBAPTDatasetHoops,
    N_REAL_AGENTS as HOOPS_N_REAL_AGENTS,
    HOOP_ID as HOOPS_ID,
)


if not torch.cuda.is_available():
    # MART has 3 hardcoded .cuda() calls inside prt.py / hrt.py for relation
    # matrices it builds on the fly. On CPU we neutralize Tensor.cuda() so
    # those tensors stay on CPU and the model runs end-to-end.
    print('CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat')
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def parse_args():
    p = argparse.ArgumentParser(description='MART on network_ml_project NBA data')
    # Existing MART-style flags
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--config', type=str, default='configs/mart_nba_pt.yaml')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--test', action='store_true')
    # Checkpoint naming: writes to ./checkpoints/<model_name>.ckpt
    p.add_argument('--model_name', type=str, required=True,
                   help='Name for the saved checkpoint at ./checkpoints/<model_name>.ckpt')
    # New: network_ml_project split
    p.add_argument('--split_path', type=str, required=False,
                   default='../network_ml_project/splits/fold0.json',
                   help='Path to network_ml_project splits/<name>.json')
    p.add_argument('--num_workers', type=int, default=4)
    # Training loss. min_ade is MART's paper-native loss; the others trade off
    # mode-diversity (min_*) vs all-modes-good (mean_*) and L2 (ade) vs squared
    # error (mse, closer to Kaggle's scoring).
    p.add_argument('--loss', type=str, default='min_ade',
                   choices=['min_ade', 'mean_ade', 'min_mse', 'mean_mse'],
                   help='Training loss: {min|mean} over K of {ADE|MSE}.')
    # Hoops augmentation: when set, swap in the 13-agent dataset that appends
    # 2 static basket nodes after the ball, bump MART_ID's entity embedding to
    # 4 classes, and mask the hoops out of loss & val metrics. Off by default;
    # the original 11-agent pipeline is unchanged.
    p.add_argument('--use_hoops', action='store_true',
                   help='Append 2 static basket-hoop nodes as extra agents.')
    # Court-mirror augmentation: training-only random reflection across the
    # court's x- and y-axes (independently, p=0.5 each). The court is centered
    # at the origin in raw feet so a reflection is a coordinate negation; under
    # z-scoring the equivalent transform is z -> -z - 2*mu/sigma along the
    # reflected axis. Off by default.
    p.add_argument('--aug_court_mirror', action='store_true',
                   help='Random court-symmetry reflection per batch sample at train time.')
    # Isotropic normalization: shared scalar std for x & y (vs per-axis). Required
    # for valid rotation augmentation and independently a win for EqMotion.
    p.add_argument('--iso_norm', action='store_true',
                   help='Isotropic (shared scalar) z-score std; needed for --aug_rotate.')
    # Rotation augmentation: random rotation about the court center, drawn from
    # U(-aug_rot_deg, aug_rot_deg). 180 = full O(2) (with --aug_court_mirror).
    # Implies --iso_norm geometrically; we enforce iso when this is > 0.
    p.add_argument('--aug_rot_deg', type=float, default=0.0,
                   help='Max rotation magnitude in degrees (0 = off; 180 = full circle).')
    # Gaussian position jitter (normed units) on the PAST only — input regularizer.
    p.add_argument('--aug_jitter', type=float, default=0.0,
                   help='Std of Gaussian jitter added to past positions (normed units).')
    # Optional overrides of capacity / schedule without editing the yaml, so a
    # single config can drive a small sweep of model sizes / lengths.
    p.add_argument('--num_epochs', type=int, default=None,
                   help='Override config num_epochs (e.g. long training).')
    p.add_argument('--dropout', type=float, default=None,
                   help='Override config dropout.')
    # Scheduler override + SGDR (warm-restart) controls. To validate the
    # schedule-shape hypothesis, run SGDR at the SAME num_epochs as the
    # single-cosine baseline so any delta is from LR shape, not extra compute.
    p.add_argument('--scheduler_type', type=str, default=None,
                   help='Override config scheduler_type '
                        '(e.g. CosineAnnealingWarmRestarts).')
    p.add_argument('--sgdr_t0', type=int, default=None,
                   help='SGDR first-cycle length in epochs (CosineAnnealingWarmRestarts).')
    p.add_argument('--sgdr_tmult', type=int, default=None,
                   help='SGDR cycle-length multiplier per restart (default 1).')
    # Logging
    p.add_argument('--hoop_feats', action='store_true',
                   help='Inject hoop-relative offset vectors as extra input features. '
                        'Requires --use_hoops (hoop positions are taken from the 2 '
                        'landmark nodes already in x_abs, so augmentation is automatic).')
    p.add_argument('--ball_dist', action='store_true',
                   help='Inject scalar Euclidean distance to the ball as an extra input '
                        'feature for every agent at every past timestep. Ball is always '
                        'at canonical index 10. Adds 1 dim to extra_input_dim.')
    p.add_argument('--edge_type_emb', action='store_true',
                   help='Add a learned edge-type embedding bias to the RT pair encoder\'s '
                        'initial edge features. Types encode basketball structure: '
                        'same-team, opponent, player↔ball, agent↔hoop, self.')
    p.add_argument('--curriculum', action='store_true',
                   help='Curriculum loss: use --loss (default min_ade) for the first '
                        '--curriculum_switch epochs, then switch to mean_mse.')
    p.add_argument('--curriculum_switch', type=int, default=None,
                   help='Epoch at which to switch from --loss to mean_mse. '
                        'Defaults to half of num_epochs if not set.')
    p.add_argument('--soft_wta_temp', type=float, default=0.5,
                   help='Temperature for soft-WTA loss (--loss soft_wta). '
                        'Lower = closer to min-of-K; higher = closer to mean-of-K.')
    p.add_argument('--laplace_nll', action='store_true',
                   help='Use Laplace NLL loss. Changes decoder output to 4D '
                        '(loc + scale). Best mode selected by ADE on loc.')
    p.add_argument('--cfi', action='store_true',
                   help='Enable Cross-modal Future Interaction decoder. Adds Branch 2 '
                        'with self-attention over K×N mode-agent tokens. '
                        'Loss = loss(branch1) + loss(branch2). Val uses branch2 mean-of-K.')
    # Logging
    p.add_argument('--wandb_project', type=str, default='NML_base')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--wandb_mode', type=str, default='online',
                   help='online | offline | disabled')
    return p.parse_args()


def _x_rel_from_x_abs(x_abs):
    """Velocity tensor in MART's convention: x_rel[t] = x_abs[t] - x_abs[t-1],
    with x_rel[0] = x_rel[1]."""
    x_rel = torch.zeros_like(x_abs)
    x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
    x_rel[:, :, 0] = x_rel[:, :, 1]
    return x_rel


def _model_forward(model, x_abs, x_rel, agent_ids, extra_feats=None, mu=None, sigma=None):
    """Dispatch to MART_ID (takes agent_ids) vs stock MART (ignores them)."""
    if isinstance(model, MART_ID):
        return model(x_abs, x_rel, agent_ids, extra_feats=extra_feats, mu=mu, sigma=sigma)
    return model(x_abs, x_rel, extra_feats=extra_feats, mu=mu, sigma=sigma)


def _compute_ball_dist_feats(x_abs):
    """Euclidean distance from every agent to the ball at every past timestep.

    Ball is always at canonical index 10 ([TeamA(5), TeamB(5), Ball]).
    Returns: [B, N, T_p, 1]
    """
    ball_pos = x_abs[:, 10:11, :, :]          # [B, 1, T_p, 2]
    dist = torch.norm(x_abs - ball_pos, dim=-1, keepdim=True)  # [B, N, T_p, 1]
    return dist


def _compute_hoop_feats(x_abs, n_real):
    """Hoop-relative offset vectors for every agent at every past timestep.

    Hoop nodes sit at x_abs indices n_real and n_real+1. Computing offsets
    from the hoop positions already in x_abs means the result is automatically
    consistent with whatever augmentation (rotation, mirror) was applied.

    Returns: [B, N, T_p, 4]  = [off_to_hoop1_x, off_to_hoop1_y, off_to_hoop2_x, off_to_hoop2_y]
    """
    hoop1 = x_abs[:, n_real:n_real+1, :, :]      # [B, 1, T_p, 2]
    hoop2 = x_abs[:, n_real+1:n_real+2, :, :]    # [B, 1, T_p, 2]
    off1 = hoop1 - x_abs                          # [B, N, T_p, 2]
    off2 = hoop2 - x_abs                          # [B, N, T_p, 2]
    return torch.cat([off1, off2], dim=-1)         # [B, N, T_p, 4]


def _augment_court(x_abs, y, center, rot_max_rad=0.0, mirror=False, jitter_std=0.0):
    """Random O(2) court-symmetry augmentation per batch instance (training only).

    Composes a per-sample random rotation and/or independent x/y reflection,
    applied *about the court center* so the transform is an exact isometry on
    the court: pairwise agent distances are preserved, MART's relation/attention
    structure stays valid, and agent_ids (team labels, ball, hoops) are
    unchanged. The SAME transform is applied to past (`x_abs`) and future (`y`)
    so the window stays coherent; `train_one_epoch` recomputes `x_rel` from the
    transformed `x_abs`, so velocities transform correctly (the constant center
    offset cancels in the time difference).

    Rotation requires ISOTROPIC normalization (--iso_norm): under anisotropic
    per-axis std a rotation in raw feet becomes a shear in the normed frame.
    Reflections are valid under either norm, but for consistency we apply both
    here in the (assumed iso) normed frame about `center`.

    Court center in normed space is `center = -mu / sigma` (raw origin (0,0) is
    the court center; z = (0 - mu) / sigma). For full rotation use
    rot_max_rad = pi; for a mild regularizer use a small angle.

    Args:
        x_abs:       [B, N, T_p, 2]  z-scored past positions
        y:           [B, N, T_f, 2]  z-scored future positions
        center:      [2]             normed court center (-mu/sigma) on device
        rot_max_rad: rotation angle drawn ~ U(-rot_max_rad, rot_max_rad). 0 = off.
        mirror:      if True, independent 50/50 x and y reflections (D2).
        jitter_std:  Gaussian position noise std (normed units) added to the
                     PAST only (input regularizer; target stays clean). 0 = off.
    Returns:
        Transformed (x_abs, y) as new tensors.
    """
    B = x_abs.shape[0]
    device = x_abs.device

    # Per-sample reflection signs (D2) folded into the linear map.
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

    # 2x2 per-sample map M = R(theta) @ diag(sx, sy), applied about `center`.
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

    y_pred: [B, N, K, T_f, 2] or [B, N, K, T_f, 4] for laplace_nll
    y_exp:  [B, N, 1, T_f, 2]

    loss_type options:
        min_ade   — min-of-K L2 (MART default, best for diversity)
        mean_ade  — mean-of-K L2 (collapses diversity, not recommended)
        min_mse   — min-of-K MSE
        mean_mse  — mean-of-K MSE (Kaggle-aligned but collapses diversity)
        soft_wta  — temperature-weighted softmin over K heads; interpolates
                    min (τ→0) and mean (τ→∞). Aligns training with mean-of-K
                    inference while preserving mode diversity.
        laplace_nll — Laplace NLL on the best mode; y_pred must be 4D
                      (loc_x, loc_y, raw_scale_x, raw_scale_y).
    """
    if loss_type == 'soft_wta':
        per_k = torch.norm(y_pred - y_exp, dim=-1).mean(dim=3)        # [B, N, K]
        weights = torch.nn.functional.softmin(per_k / soft_wta_temp, dim=2).detach()
        return (weights * per_k).sum(dim=2).mean()

    if loss_type == 'laplace_nll':
        MIN_SCALE = 1e-3
        loc   = y_pred[..., :2]                                        # [B, N, K, T_f, 2]
        scale = torch.nn.functional.softplus(y_pred[..., 2:]) + MIN_SCALE  # [B, N, K, T_f, 2]
        # Best mode by ADE on loc
        per_k = torch.norm(loc - y_exp, dim=-1).mean(dim=3)           # [B, N, K]
        best_k = per_k.argmin(dim=2)                                   # [B, N]
        B, N, K, T_f, _ = loc.shape
        idx = best_k.view(B, N, 1, 1, 1).expand(B, N, 1, T_f, 2)
        best_loc   = loc.gather(2, idx).squeeze(2)                     # [B, N, T_f, 2]
        best_scale = scale.gather(2, idx).squeeze(2)                   # [B, N, T_f, 2]
        y_gt = y_exp.squeeze(2)                                        # [B, N, T_f, 2]
        nll = torch.log(2 * best_scale) + torch.abs(y_gt - best_loc) / best_scale
        return nll.mean()

    if loss_type in ('min_ade', 'mean_ade'):
        per_k = torch.norm(y_pred - y_exp, dim=-1).mean(dim=3)        # [B, N, K]
    elif loss_type in ('min_mse', 'mean_mse'):
        per_k = ((y_pred - y_exp) ** 2).mean(dim=(3, 4))              # [B, N, K]
    else:
        raise ValueError(f'unknown loss_type: {loss_type}')

    if loss_type.startswith('min_'):
        reduced = per_k.min(dim=2)[0]   # [B, N]
    else:
        reduced = per_k.mean(dim=2)     # [B, N]

    return reduced.mean()


def train_one_epoch(epoch, model, optimizer, loader, opts, device, aug=None, mu=None, sigma=None):
    """One training epoch.

    aug: when geometric augmentation is enabled, a dict with keys
        {center, rot_max_rad, mirror, jitter_std} forwarded to _augment_court.
        None disables augmentation entirely (default).
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
                x_abs, y, aug['center'],
                rot_max_rad=aug['rot_max_rad'],
                mirror=aug['mirror'],
                jitter_std=aug['jitter_std'],
            )

        extra_parts = []
        if opts.get('hoop_feats', False):
            extra_parts.append(_compute_hoop_feats(x_abs, HOOPS_N_REAL_AGENTS))
        if opts.get('ball_dist', False):
            extra_parts.append(_compute_ball_dist_feats(x_abs))
        extra_feats = torch.cat(extra_parts, dim=-1) if extra_parts else None

        x_rel = _x_rel_from_x_abs(x_abs)
        fwd = _model_forward(model, x_abs, x_rel, agent_ids, extra_feats, mu, sigma)

        # CFI returns (loc1, loc2); standard returns a single tensor
        cfi_mode = isinstance(fwd, tuple)
        y_pred = fwd[1] if cfi_mode else fwd    # use branch2 as primary prediction

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

        y_exp = y[:, :, None, :, :]         # [B, N, 1, T_f, 2]

        if opts.get('use_hoops', False):
            y_pred_loss = y_pred[:, :HOOPS_N_REAL_AGENTS]
            y_exp_loss = y_exp[:, :HOOPS_N_REAL_AGENTS]
        else:
            y_pred_loss = y_pred
            y_exp_loss = y_exp

        # Curriculum: switch loss type after curriculum_switch epoch
        effective_loss = opts.loss
        if opts.get('curriculum', False) and epoch >= opts.get('curriculum_switch', opts.num_epochs // 2):
            effective_loss = 'mean_mse'

        loss = compute_loss(y_pred_loss, y_exp_loss, effective_loss,
                            soft_wta_temp=opts.get('soft_wta_temp', 0.5))

        # CFI: add branch1 loss to encourage diverse hypotheses before refinement
        if cfi_mode:
            loc1 = fwd[0]
            loc1_loss = loc1[:, :HOOPS_N_REAL_AGENTS] if opts.get('use_hoops', False) else loc1
            loss = loss + compute_loss(loc1_loss, y_exp_loss, effective_loss,
                                       soft_wta_temp=opts.get('soft_wta_temp', 0.5))

        optimizer.zero_grad()
        loss.backward()
        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
        optimizer.step()

        # Weight by the number of agents the loss was actually averaged over.
        n_scored = HOOPS_N_REAL_AGENTS if opts.get('use_hoops', False) else N
        total_loss += loss.item() * B * n_scored
        total_n += B * n_scored

        if i % 100 == 0:
            th = get_th(opts, model)
            print(
                f'[TRAIN] Epoch {epoch + 1:03d}/{opts.num_epochs:03d} | '
                f'It {i + 1:04d}/{n_iter:04d} | '
                f'Loss {loss.item():.4f} | th {th} | '
                f'lr {optimizer.param_groups[0]["lr"]:.2e}'
            )

    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_minADE_minFDE(model, loader, opts, device, split_name, mu=None, sigma=None):
    """Validation metrics.

    Reports:
      * `val/mse_ft` — mean squared error in feet², over real entities (11),
        on the MEAN-of-K prediction. Matches Kaggle's single-shot mean MSE
        scoring AND EqMotion's `val/mse_ft` definition exactly, so numbers
        are directly comparable across architectures. Requires `mu`/`sigma`
        (the train-split z-score stats) for denormalization.
      * `minADE` / `minFDE` — min-of-K ADE/FDE in **normalized units** (z-score
        space). Kept for parity with MART's paper-native loss; NOT comparable
        to Kaggle or to EqMotion's feet-space metric.

    Uses whatever sampler the caller wires up (we wire `WindowEvalSampler`
    for deterministic multi-window evaluation, matching EqMotion's full-val).
    """
    model.eval()
    sum_ade = 0.0
    sum_fde = 0.0
    sum_loss = 0.0
    n_total = 0
    sse_ft = 0.0   # sum of squared errors, in feet², on real entities only
    cnt_ft = 0     # element count for the mean below

    use_ft = mu is not None and sigma is not None
    if use_ft:
        # Broadcast as [1, 1, 1, 2] over (B, N, T_f, 2) for the mean-of-K pred.
        mu_b = mu.to(device).view(1, 1, 1, 2).float()
        sigma_b = sigma.to(device).view(1, 1, 1, 2).float()

    for x_abs, y, agent_ids in loader:
        x_abs = x_abs.to(device)
        y = y.to(device)
        agent_ids = agent_ids.to(device)
        B, N, _, _ = x_abs.shape

        extra_parts = []
        if opts.get('hoop_feats', False):
            extra_parts.append(_compute_hoop_feats(x_abs, HOOPS_N_REAL_AGENTS))
        if opts.get('ball_dist', False):
            extra_parts.append(_compute_ball_dist_feats(x_abs))
        extra_feats = torch.cat(extra_parts, dim=-1) if extra_parts else None

        x_rel = _x_rel_from_x_abs(x_abs)
        fwd = _model_forward(model, x_abs, x_rel, agent_ids, extra_feats, mu, sigma)
        y_pred = fwd[1] if isinstance(fwd, tuple) else fwd   # [B, N, K, T_f, 2 or 4]

        # Laplace NLL: strip scale channels — only loc matters for metrics
        if y_pred.shape[-1] == 4:
            y_pred = y_pred[..., :2]

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

        y_exp = y[:, :, None, :, :]        # [B, N, 1, T_f, 2]

        # Mask hoops out of all val metrics — they're static so they trivially
        # zero the loss and inflate apparent quality (this exact bug bit
        # EqMotion: val 2.82 → real 3.33 once hoops were stripped).
        if opts.get('use_hoops', False):
            y_pred = y_pred[:, :HOOPS_N_REAL_AGENTS]
            y_exp = y_exp[:, :HOOPS_N_REAL_AGENTS]
            y_real = y[:, :HOOPS_N_REAL_AGENTS]
            N_scored = HOOPS_N_REAL_AGENTS
        else:
            y_real = y
            N_scored = N

        # min-of-K ADE / FDE (z-score space, kept for parity with paper).
        per_step_err = torch.norm(y_pred - y_exp, dim=-1)            # [B, N, K, T_f]
        ade_perK = per_step_err.mean(dim=3)                          # [B, N, K]
        min_ade = ade_perK.min(dim=2)[0]                             # [B, N]
        fde_perK = per_step_err[:, :, :, -1]                         # [B, N, K]
        min_fde = fde_perK.min(dim=2)[0]                             # [B, N]
        loss_val = min_ade.mean()
        sum_ade += min_ade.sum().item()
        sum_fde += min_fde.sum().item()
        sum_loss += loss_val.item() * B * N_scored
        n_total += B * N_scored

        # val/mse_ft on the MEAN-of-K prediction, denormalized to feet.
        # Equivalent to what submit_nba_pt.py writes (--reduce mean), so val
        # tracks Kaggle directly. Matches EqMotion's compute_mse exactly:
        # mean over (T_f, B, N_real, 2) of squared error in feet².
        if use_ft:
            y_mean = y_pred.mean(dim=2)                              # [B, N_real, T_f, 2]
            y_mean_ft = y_mean * sigma_b + mu_b
            y_tgt_ft = y_real * sigma_b + mu_b
            sse_ft += ((y_mean_ft - y_tgt_ft) ** 2).sum().item()
            cnt_ft += y_mean_ft.numel()

    avg_ade = sum_ade / max(n_total, 1)
    avg_fde = sum_fde / max(n_total, 1)
    avg_loss = sum_loss / max(n_total, 1)
    mse_ft = sse_ft / cnt_ft if use_ft and cnt_ft > 0 else float('nan')

    if use_ft:
        print(
            f'[{split_name.upper()}] mse_ft = {mse_ft:.4f} ft²  '
            f'| minADE/minFDE (z-score) @ {opts.future_length} steps: '
            f'{avg_ade:.4f} / {avg_fde:.4f}'
        )
    else:
        print(
            f'[{split_name.upper()}] minADE/minFDE @ {opts.future_length} steps: '
            f'{avg_ade:.4f} / {avg_fde:.4f}'
        )
    return {'loss': avg_loss, 'minADE': avg_ade, 'minFDE': avg_fde, 'mse_ft': mse_ft}


def main():
    args = parse_args()
    setup_seed(args.seed)
    print('[INFO] seed:', args.seed)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] device:', device)

    opts = load_config(args.config)
    opts.loss = args.loss
    opts.use_hoops = args.use_hoops
    opts.hoop_feats = args.hoop_feats
    opts.ball_dist = args.ball_dist
    opts.edge_type_emb = args.edge_type_emb
    opts.cfi = args.cfi
    opts.curriculum = args.curriculum
    opts.curriculum_switch = args.curriculum_switch   # resolved to int in main()
    opts.soft_wta_temp = args.soft_wta_temp
    opts.laplace_nll = args.laplace_nll
    if args.laplace_nll:
        opts.loss = 'laplace_nll'
    opts.extra_input_dim = (4 if args.hoop_feats else 0) + (1 if args.ball_dist else 0)
    opts.aug_court_mirror = args.aug_court_mirror
    opts.iso_norm = args.iso_norm
    opts.aug_rot_deg = args.aug_rot_deg
    opts.aug_jitter = args.aug_jitter
    if args.num_epochs is not None:
        opts.num_epochs = args.num_epochs
    if args.dropout is not None:
        opts.dropout = args.dropout
    if args.scheduler_type is not None:
        opts.scheduler_type = args.scheduler_type
    if args.sgdr_t0 is not None:
        opts.sgdr_t0 = args.sgdr_t0
    if args.sgdr_tmult is not None:
        opts.sgdr_tmult = args.sgdr_tmult

    # Rotation in the normed frame is only an isometry under isotropic scaling.
    # Force iso on whenever rotation is requested so we never silently train on
    # sheared (geometrically invalid) augmented samples.
    if opts.aug_rot_deg > 0.0 and not opts.iso_norm:
        print('[WARN] --aug_rot_deg > 0 requires isotropic norm; enabling --iso_norm.')
        opts.iso_norm = True

    if opts.hoop_feats and not opts.use_hoops:
        raise ValueError('--hoop_feats requires --use_hoops (hoop positions come from the landmark nodes in x_abs).')

    # Resolve curriculum switch epoch now that num_epochs is finalised
    if opts.curriculum and opts.curriculum_switch is None:
        opts.curriculum_switch = opts.num_epochs // 2
    if opts.curriculum:
        print(f'[INFO] curriculum: {opts.loss} for epochs 0–{opts.curriculum_switch-1}, '
              f'then mean_mse for epochs {opts.curriculum_switch}–{opts.num_epochs-1}')

    print(f'[INFO] training loss: {opts.loss}')
    print(f'[INFO] use_hoops: {opts.use_hoops}')
    print(f'[INFO] hoop_feats: {opts.hoop_feats}  ball_dist: {opts.ball_dist}  '
          f'extra_input_dim={opts.extra_input_dim}')
    print(f'[INFO] iso_norm: {opts.iso_norm}')
    print(f'[INFO] aug: mirror={opts.aug_court_mirror} rot_deg={opts.aug_rot_deg} '
          f'jitter={opts.aug_jitter}')
    print('[INFO] opts:', opts)

    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.model_name}.ckpt')
    print(f'[INFO] final-epoch checkpoint will be saved to: {ckpt_path}')

    # In --test mode the architecture must match the checkpoint exactly. Trust
    # the checkpoint's saved use_hoops over the CLI flag so a stale --use_hoops
    # doesn't silently build the model with the wrong embedding size and crash
    # state_dict load. (Training mode uses the CLI flag as authored.)
    if args.test and os.path.isfile(ckpt_path):
        ckpt_opts = torch.load(
            ckpt_path, map_location='cpu', weights_only=False,
        ).get('opts', {})
        ckpt_uh = bool(ckpt_opts.get('use_hoops', False))
        if ckpt_uh != opts.use_hoops:
            print(
                f'[INFO] --test: overriding --use_hoops={opts.use_hoops} '
                f'with checkpoint value {ckpt_uh}'
            )
            opts.use_hoops = ckpt_uh

    # ---- Data ----
    train_files, val_files = load_split_files(args.split_path)
    print(f'[INFO] split: {len(train_files)} train, {len(val_files)} val')

    mu, sigma = compute_xy_stats(train_files, iso=opts.iso_norm)
    print(f'[INFO] norm stats (iso={opts.iso_norm}): mu={mu.tolist()}, sigma={sigma.tolist()}')

    # Build the geometric-augmentation spec. The court center in normed space is
    # -mu/sigma (raw origin (0,0) is the court center). Rotation/reflection are
    # applied about it so distances are preserved. None disables augmentation.
    if opts.aug_court_mirror or opts.aug_rot_deg > 0.0 or opts.aug_jitter > 0.0:
        aug = {
            'center': (-mu / sigma).to(device),
            'rot_max_rad': math.radians(opts.aug_rot_deg),
            'mirror': bool(opts.aug_court_mirror),
            'jitter_std': float(opts.aug_jitter),
        }
        print(f'[INFO] augmentation enabled: center={aug["center"].tolist()} '
              f'rot_max_rad={aug["rot_max_rad"]:.3f} mirror={aug["mirror"]} '
              f'jitter={aug["jitter_std"]}')
    else:
        aug = None

    DatasetCls = MARTNBAPTDatasetHoops if opts.use_hoops else MARTNBAPTDataset
    train_set = DatasetCls(
        train_files, mu, sigma, opts.past_length, opts.future_length,
    )
    val_set = DatasetCls(
        val_files, mu, sigma, opts.past_length, opts.future_length,
    )
    print(f'[INFO] usable sequences: train={len(train_set)}, val={len(val_set)}')

    train_sampler = WindowSampler(
        opts.batch_size, train_set.max_start, seed=args.seed, shuffle=True,
    )
    # Deterministic multi-window val sampler so val/mse_ft is stable across
    # epochs and directly comparable to EqMotion's --full-val numbers.
    val_sampler = WindowEvalSampler(val_set.max_start, windows_per_seq=8)
    print(f'[INFO] val windows: {len(val_sampler)} (deterministic 8/seq)')

    train_loader = DataLoader(
        train_set, batch_size=opts.batch_size, sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set, batch_size=opts.batch_size, sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Model ----
    use_id = bool(opts.get('use_entity_embed', False))
    if use_id:
        # Hoops add a new entity class (id=3 = basket) on top of the default
        # 3 classes (TeamA, TeamB, Ball). Bump num_entity_types accordingly so
        # the nn.Embedding has a row for it; without this MART_ID would index
        # out-of-bounds on the hoops loader's agent_ids.
        n_entity_types = (HOOPS_ID + 1) if opts.use_hoops else 3
        print(
            f'[INFO] using MART_ID with embed_dim={opts.embed_dim}, '
            f'num_entity_types={n_entity_types}'
        )
        model = MART_ID(opts, num_entity_types=n_entity_types).to(device)
    else:
        print('[INFO] using stock MART (no entity embedding)')
        model = MART(opts).to(device)
    print(model)
    print(f'[INFO] params: {sum(p.numel() for p in model.parameters())}')

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)
    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.decay_step, gamma=opts.decay_gamma,
        )
    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=opts.milestones, gamma=opts.decay_gamma,
        )
    elif opts.scheduler_type == 'CosineAnnealingLR':
        # Same shape as EqMotion (eta_min = lr*0.02) so cross-arch comparisons
        # share an LR schedule. T_max=num_epochs anneals over the full run.
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opts.num_epochs, eta_min=opts.lr * 0.02,
        )
    elif opts.scheduler_type == 'CosineAnnealingWarmRestarts':
        # SGDR: cosine annealing that periodically RESTARTS the LR back to its
        # peak. First cycle is sgdr_t0 epochs; each subsequent cycle is sgdr_tmult
        # times longer. eta_min matches CosineAnnealingLR for a fair comparison.
        # NOTE: step() must be called once per epoch (we do, after each epoch),
        # so the cycle lengths are in EPOCH units. Budget-match num_epochs to the
        # single-cosine baseline so any delta is attributable to LR shape, not
        # extra compute.
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=opts.sgdr_t0, T_mult=opts.get('sgdr_tmult', 1),
            eta_min=opts.lr * 0.02,
        )
    else:
        scheduler = None

    # ---- Test-only path ----
    if args.test:
        print(f'[INFO] Loading model from: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        eval_minADE_minFDE(
            model, val_loader, opts, device, split_name='val',
            mu=mu, sigma=sigma,
        )
        return

    # ---- Train ----
    # We save BOTH the final-epoch checkpoint (ckpt_path) and the best-by-val
    # checkpoint (best_ckpt_path). With long cosine schedules the last epoch is
    # usually near-best, but best-tracking guards against late-run drift.
    val_metrics = None
    best_ckpt_path = os.path.join(ckpt_dir, f'{args.model_name}_best.ckpt')
    best_mse = float('inf')
    # For SGDR: snapshot the model at the END of each cosine cycle (its minimum),
    # detected by the LR jumping back up after scheduler.step(). These per-cycle
    # snapshots form a free same-arch "snapshot ensemble" — a different lever from
    # the schedule-shape question, so we capture it regardless.
    is_sgdr = (opts.scheduler_type == 'CosineAnnealingWarmRestarts')
    snapshot_dir = os.path.join(ckpt_dir, f'{args.model_name}_snapshots')
    if is_sgdr:
        os.makedirs(snapshot_dir, exist_ok=True)
    n_snapshots = 0

    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={**dict(opts), 'split_path': args.split_path, 'seed': args.seed},
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    ):
        for epoch in range(opts.num_epochs):
            train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                epoch, model, optimizer, train_loader, opts, device, aug,
                mu=mu, sigma=sigma,
            )
            val_metrics = eval_minADE_minFDE(
                model, val_loader, opts, device, split_name='val',
                mu=mu, sigma=sigma,
            )

            lr_before = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
            lr_after = optimizer.param_groups[0]['lr']

            # SGDR restart detection: within a cosine cycle the LR is monotone
            # DEcreasing, so ANY increase after step() is unambiguously a restart.
            # (Using a strict >2x gate misfires at tiny T_0 where the pre-restart
            # LR is the cosine midpoint, not eta_min; a 1.1x margin is robust to
            # bottom-of-cosine float jitter while still catching every restart.)
            # The model state *before* stepping is that cycle's minimum, so
            # snapshot it now (we haven't mutated weights between step() calls).
            if is_sgdr and lr_after > lr_before * 1.1:
                snap_path = os.path.join(snapshot_dir, f'cycle_{n_snapshots:02d}.ckpt')
                torch.save(
                    {
                        'epoch': epoch, 'state_dict': model.state_dict(),
                        'opts': dict(opts), 'mu': mu, 'sigma': sigma,
                        'val_mse_ft': val_metrics['mse_ft'],
                    },
                    snap_path,
                )
                print(f'[SGDR] cycle {n_snapshots} ended at epoch {epoch} '
                      f'(val/mse_ft {val_metrics["mse_ft"]:.4f}) -> {snap_path}')
                n_snapshots += 1

            if val_metrics['mse_ft'] < best_mse:
                best_mse = val_metrics['mse_ft']
                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'opts': dict(opts),
                        'mu': mu,
                        'sigma': sigma,
                        'val_mse_ft': val_metrics['mse_ft'],
                        'val_minADE': val_metrics['minADE'],
                        'val_minFDE': val_metrics['minFDE'],
                    },
                    best_ckpt_path,
                )

            log = {
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'th': get_th(opts, model),
                'train/loss': train_loss,
                'val/loss': val_metrics['loss'],
                'val/mse_ft': val_metrics['mse_ft'],
                'val/minADE': val_metrics['minADE'],
                'val/minFDE': val_metrics['minFDE'],
            }
            wandb.log(log)

            print(
                f'[INFO] Epoch {epoch:03d} done | train_loss {train_loss:.4f} | '
                f'val/mse_ft {val_metrics["mse_ft"]:.4f} ft² | '
                f'val_minADE {val_metrics["minADE"]:.4f}'
            )

        # ---- After all epochs: save the final checkpoint and report final val ----
        # Single artifact at ckpt_path; everything needed to rebuild the model
        # and run submit_nba_pt.py / eval.py lives inside.
        torch.save(
            {
                'epoch': opts.num_epochs - 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'opts': dict(opts),
                'mu': mu,
                'sigma': sigma,
                'val_minADE': val_metrics['minADE'],
                'val_minFDE': val_metrics['minFDE'],
                'val_mse_ft': val_metrics['mse_ft'],
            },
            ckpt_path,
        )
        print(f'\n[INFO] Training complete. Saved final-epoch checkpoint to {ckpt_path}')
        print(f'[INFO] Best-by-val checkpoint: {best_ckpt_path} (val/mse_ft {best_mse:.4f})')
        print('=== Final val metrics (last epoch) ===')
        print(f'  val/mse_ft : {val_metrics["mse_ft"]:.4f} ft²  (Kaggle-comparable)')
        print(f'  val/minADE : {val_metrics["minADE"]:.4f}  (normalized, min-of-K)')
        print(f'  val/minFDE : {val_metrics["minFDE"]:.4f}  (normalized, min-of-K)')
        print(f'  val/loss   : {val_metrics["loss"]:.4f}')
        print(
            '[INFO] For metrics in feet (denormalized, ADE/FDE/MSE + min-of-K '
            'oracle), run eval.py against the saved checkpoint.'
        )


if __name__ == '__main__':
    main()
