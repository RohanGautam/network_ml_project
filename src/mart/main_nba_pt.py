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


def _model_forward(model, x_abs, x_rel, agent_ids):
    """Dispatch to MART_ID (takes agent_ids) vs stock MART (ignores them)."""
    if isinstance(model, MART_ID):
        return model(x_abs, x_rel, agent_ids)
    return model(x_abs, x_rel)


def _augment_court_mirror(x_abs, y, aug_offset):
    """Random court-symmetry reflections per batch instance (training only).

    Raw coords are court-centered, so a court reflection is a coordinate
    negation in raw feet. Under z-scoring z = (raw - mu) / sigma, the
    equivalent transform along axis c is z -> -z - 2*mu[c]/sigma[c]; we
    precompute aug_offset = 2 * mu / sigma so the inner loop is a plain
    `-z - aug_offset[c]` per flipped axis.

    Reflections are isometries: pairwise agent distances are preserved, so
    MART's relation/attention structure stays valid and agent_ids (team
    labels, ball, hoops) are unchanged — team identity is independent of
    which side of the court a team is playing on. Velocity is *not* mutated
    here; train_one_epoch recomputes x_rel from the reflected x_abs, which
    naturally flips the velocity sign (the constant offset cancels in the
    time difference).

    Args:
        x_abs:      [B, N, T_p, 2]  z-scored past positions
        y:          [B, N, T_f, 2]  z-scored future positions
        aug_offset: [2]             precomputed 2 * mu / sigma on x_abs.device
    Returns:
        Reflected (x_abs, y) as new tensors.
    """
    B = x_abs.shape[0]
    device = x_abs.device
    flip_x = (torch.rand(B, device=device) < 0.5).view(B, 1, 1)  # negate x coord
    flip_y = (torch.rand(B, device=device) < 0.5).view(B, 1, 1)  # negate y coord

    x_abs = x_abs.clone()
    y = y.clone()

    # Channel 0 = court-length axis (x): swap left/right basket sides.
    x_abs[..., 0] = torch.where(flip_x, -x_abs[..., 0] - aug_offset[0], x_abs[..., 0])
    y[..., 0]     = torch.where(flip_x, -y[..., 0]     - aug_offset[0], y[..., 0])

    # Channel 1 = court-width axis (y): swap top/bottom sideline.
    x_abs[..., 1] = torch.where(flip_y, -x_abs[..., 1] - aug_offset[1], x_abs[..., 1])
    y[..., 1]     = torch.where(flip_y, -y[..., 1]     - aug_offset[1], y[..., 1])

    return x_abs, y


def compute_loss(y_pred, y_exp, loss_type):
    """Aggregate K hypotheses into a scalar loss.

    y_pred: [B, N, K, T_f, 2]   y_exp: [B, N, 1, T_f, 2]

    Per-K error:
        ade  -> mean L2 distance over T_f
        mse  -> mean squared error over T_f and the (x, y) axes
    Aggregation over K:
        min  -> best-of-K (encourages mode diversity, MART's default)
        mean -> all-of-K (pushes every head toward GT, collapses diversity)
    """
    if loss_type in ('min_ade', 'mean_ade'):
        per_k = torch.norm(y_pred - y_exp, dim=-1).mean(dim=3)   # [B, N, K]
    elif loss_type in ('min_mse', 'mean_mse'):
        per_k = ((y_pred - y_exp) ** 2).mean(dim=(3, 4))         # [B, N, K]
    else:
        raise ValueError(f'unknown loss_type: {loss_type}')

    if loss_type.startswith('min_'):
        reduced = per_k.min(dim=2)[0]   # [B, N]
    else:
        reduced = per_k.mean(dim=2)     # [B, N]

    return reduced.mean()


def train_one_epoch(epoch, model, optimizer, loader, opts, device, aug_offset=None):
    """One training epoch.

    aug_offset: when --aug_court_mirror is set, the precomputed [2] tensor
        2 * mu / sigma on `device`, used by _augment_court_mirror. None
        disables the augmentation entirely (default).
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

        if aug_offset is not None:
            x_abs, y = _augment_court_mirror(x_abs, y, aug_offset)

        x_rel = _x_rel_from_x_abs(x_abs)
        y_pred = _model_forward(model, x_abs, x_rel, agent_ids)   # [B, N, K, T_f, 2]

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

        y_exp = y[:, :, None, :, :]         # [B, N, 1, T_f, 2]

        # Hoops are static; including them in the loss trivially shrinks it and
        # dilutes the gradient away from the real 11 entities. Slice them off
        # before reduction. Hoops are always appended at the tail in the
        # canonical layout, so [:N_real] is exactly the players + ball.
        if opts.get('use_hoops', False):
            y_pred_loss = y_pred[:, :HOOPS_N_REAL_AGENTS]
            y_exp_loss = y_exp[:, :HOOPS_N_REAL_AGENTS]
        else:
            y_pred_loss = y_pred
            y_exp_loss = y_exp
        loss = compute_loss(y_pred_loss, y_exp_loss, opts.loss)

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

        x_rel = _x_rel_from_x_abs(x_abs)
        y_pred = _model_forward(model, x_abs, x_rel, agent_ids)   # [B, N, K, T_f, 2]

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
    opts.aug_court_mirror = args.aug_court_mirror
    print(f'[INFO] training loss: {opts.loss}')
    print(f'[INFO] use_hoops: {opts.use_hoops}')
    print(f'[INFO] aug_court_mirror: {opts.aug_court_mirror}')
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

    mu, sigma = compute_xy_stats(train_files)
    print(f'[INFO] norm stats: mu={mu.tolist()}, sigma={sigma.tolist()}')

    # Precompute the z-scored-space reflection offset once. mu is non-zero in
    # general (empirical train-split mean isn't exactly the court center), so a
    # naive negation would shift the reflected court; -z - aug_offset reflects
    # around the true court origin. None disables the augmentation entirely.
    if opts.aug_court_mirror:
        aug_offset = (2.0 * mu / sigma).to(device)
        print(f'[INFO] aug_court_mirror offset (2*mu/sigma): {aug_offset.tolist()}')
    else:
        aug_offset = None

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
    # We checkpoint only after the final epoch (no best-on-val tracking). Val
    # metrics are still computed every epoch for the wandb curve and the per-
    # epoch log line, but they do not gate any save.
    val_metrics = None

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
                epoch, model, optimizer, train_loader, opts, device, aug_offset,
            )
            val_metrics = eval_minADE_minFDE(
                model, val_loader, opts, device, split_name='val',
                mu=mu, sigma=sigma,
            )

            if scheduler is not None:
                scheduler.step()

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
