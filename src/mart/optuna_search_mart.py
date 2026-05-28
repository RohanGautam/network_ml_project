"""Optuna hyperparameter search for MART / MART_ID on network_ml_project NBA data.

Two-stage workflow:
    1. SEARCH: --n_trials trials (default 40). Each trial is a *full*
       --trial_epochs run (default 80) — no pruning, no intermediate
       stopping. Trials are ranked by the *last-epoch* val mean-of-K ADE
       (the same quantity --loss mean_ade optimizes), so HPs that converge
       well over the full budget beat HPs that happen to spike early.
    2. FINAL RETRAIN: with the winning HPs, train from scratch for
       --final_epochs (default 100) and save the last-epoch checkpoint to
           checkpoints/<final_model_name>.ckpt
       in the exact format main_nba_pt.py writes — submit_nba_pt.py and
       eval.py both consume it directly.

Per-trial *and* final-retrain fixed settings (forced via apply_search_params):
    - loss              = 'mean_ade'   (mean over K of mean L2 over T_f)
    - use_hoops         = False        (no basket landmark nodes)
    - aug_court_mirror  = False        (no court-reflection augmentation)
Optuna only varies the architecture / optimization knobs below.

Why "last epoch" instead of "best epoch" for trial ranking:
    Picking by best-on-val while *also* tuning HPs on val double-dips the
    val signal and tends to crown trials that got lucky on one epoch.
    Comparing by the last epoch rewards converged performance.

No pruner is used: with last-epoch comparison, killing a trial early on
intermediate scores would discard runs that improve late.

Search space (knobs with the highest impact on val mean ADE):
    - lr                  : 1e-4 .. 1e-3, log-uniform
    - weight_decay        : 1e-10 .. 1e-4, log-uniform   (Adam L2 reg)
    - dropout             : 0.0, 0.1, 0.2
    - num_layers          : 2, 3, 4         (RT + HRT stack depth)
    - model_dim           : 32, 64, 128     (num_heads=8 divides all)
    - hidden_dim          : 64, 128, 256
    - decoder_hidden_dim  : 64, 128, 256
NOT searched:
    - sample_k, num_heads, past/future_length: fixed (architectural / horizon).
    - embed_dim: only meaningful when use_entity_embed=true; kept at yaml default.
    - scheduler params: kept at yaml default (decay_step/gamma or milestones).

Example:
    python optuna_search_mart.py \\
        --config configs/mart_nba_pt.yaml \\
        --split_path ../network_ml_project/splits/fold0.json \\
        --n_trials 40 --trial_epochs 80 --final_epochs 100 \\
        --study_name mart_nba_meanade \\
        --final_model_name mart_meanade_best --gpu 0
"""

import argparse
import copy
import math
import os
import sys

sys.path.append(os.getcwd())

os.environ.setdefault(
    'WANDB_API_KEY',
    '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',
)

import optuna
import torch
import wandb

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


if not torch.cuda.is_available():
    print('CUDA not available -> patching Tensor.cuda() as a no-op for CPU compat')
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


from utils import load_config, setup_seed
from models.mart import MART
from models.mart_id import MART_ID
from loaders.dataloader_nba_pt import (
    MARTNBAPTDataset,
    WindowSampler,
    compute_xy_stats,
    load_split_files,
)
# Reuse the exact same train step as main_nba_pt.py so trials see identical
# gradient updates as a real training run. _x_rel_from_x_abs / _model_forward
# are imported because our local val helper computes both min-of-K and
# mean-of-K ADE in one pass (the stock eval_minADE_minFDE only returns min).
from main_nba_pt import (
    train_one_epoch,
    _x_rel_from_x_abs,
    _model_forward,
)


# ----------------------------- CLI -----------------------------

def parse_cli():
    p = argparse.ArgumentParser(
        description='Optuna search for MART (mean ADE, no aug, no hoops)',
    )
    p.add_argument('--config', type=str, default='configs/mart_nba_pt.yaml',
                   help='Base yaml; provides all fixed knobs (past/future, '
                        'use_entity_embed, batch_size, scheduler_type, ...).')
    p.add_argument('--split_path', type=str, required=True,
                   help='Path to network_ml_project splits/<name>.json')
    # Search budget
    p.add_argument('--n_trials', type=int, default=40,
                   help='Total HP configurations to try (default: 40).')
    p.add_argument('--trial_epochs', type=int, default=80,
                   help='Epochs per trial (default: 80). Each trial runs to '
                        'completion; last-epoch val is the comparison metric.')
    p.add_argument('--final_epochs', type=int, default=100,
                   help='Epochs for the final retrain of the winning HPs '
                        '(default: 100). Run from scratch after the search.')
    # Runtime
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    # Outputs
    p.add_argument('--study_name', type=str, default='mart_nba_meanade',
                   help="Optuna study name. Also used as the sqlite stem when "
                        "--storage is given and as the trial run-name prefix.")
    p.add_argument('--final_model_name', type=str, required=True,
                   help='Path label for the final retrain checkpoint. A bare '
                        'name (e.g. mart_best) writes to '
                        'checkpoints/<final_model_name>.ckpt; an absolute path '
                        '(e.g. /scratch/me/mart_best) writes to '
                        '<final_model_name>.ckpt directly. Do not include the '
                        '.ckpt extension.')
    p.add_argument('--storage', type=str, default=None,
                   help='Optional sqlite path for study persistence, e.g. '
                        'sqlite:///mart_meanade.db (lets you resume after a crash).')
    # Logging
    p.add_argument('--wandb_project', type=str, default='mart_nba_optuna')
    p.add_argument('--wandb_mode', type=str, default='online',
                   help='online | offline | disabled')
    return p.parse_args()


# ------------------------ Search space ------------------------

def suggest_search_params(trial):
    """The 7 knobs with the highest impact on val mean ADE.

    For mean-of-K ADE the K decoder heads collapse toward a single point
    estimate (all heads pulled to GT together), so the most-impactful axes are
    optimization (lr, weight_decay), regularization (dropout), and a modest
    amount of capacity (depth + widths). Diversity-oriented knobs (sample_k)
    don't help and inflate compute, so they're held fixed.
    """
    return {
        'lr':                 trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'weight_decay':       trial.suggest_float('weight_decay', 1e-10, 1e-4, log=True),
        'dropout':            trial.suggest_categorical('dropout', [0.0, 0.1, 0.2]),
        'num_layers':         trial.suggest_categorical('num_layers', [2, 3, 4]),
        'model_dim':          trial.suggest_categorical('model_dim', [32, 64, 128]),
        'hidden_dim':         trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'decoder_hidden_dim': trial.suggest_categorical('decoder_hidden_dim', [64, 128, 256]),
    }


def apply_search_params(base_opts, params):
    """Return a fresh Box opts with `params` applied + fixed sweep-wide knobs."""
    opts = copy.deepcopy(base_opts)
    for k, v in params.items():
        opts[k] = v
    # Sweep-wide constants. These three are forced regardless of what the
    # base yaml says, per the user spec for this study.
    opts.loss = 'mean_ade'
    opts.use_hoops = False
    opts.aug_court_mirror = False
    return opts


# ------------------------ Model + data ------------------------

def build_model(opts, device):
    use_id = bool(opts.get('use_entity_embed', False))
    if use_id:
        # Sweep forces use_hoops=False, so the entity embedding stays at the
        # default 3 classes (TeamA, TeamB, Ball).
        return MART_ID(opts).to(device)
    return MART(opts).to(device)


def build_loaders(opts, mu, sigma, split_path, num_workers, seed):
    train_files, val_files = load_split_files(split_path)
    train_set = MARTNBAPTDataset(train_files, mu, sigma,
                                 opts.past_length, opts.future_length)
    val_set = MARTNBAPTDataset(val_files, mu, sigma,
                               opts.past_length, opts.future_length)
    train_sampler = WindowSampler(opts.batch_size, train_set.max_start,
                                  seed=seed, shuffle=True)
    val_sampler = WindowSampler(opts.batch_size, val_set.max_start,
                                seed=seed, shuffle=False)
    train_loader = DataLoader(
        train_set, batch_size=opts.batch_size, sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set, batch_size=opts.batch_size, sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler


def _make_scheduler(opts, optimizer):
    if opts.scheduler_type == 'StepLR':
        return lr_scheduler.StepLR(optimizer,
                                   step_size=opts.decay_step,
                                   gamma=opts.decay_gamma)
    if opts.scheduler_type == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer,
                                        milestones=opts.milestones,
                                        gamma=opts.decay_gamma)
    return None


# ------------------------ Val helper --------------------------

@torch.no_grad()
def eval_val_metrics(model, loader, opts, device):
    """Val metrics in normalized units: min-of-K and mean-of-K ADE, min FDE.

    Returns the same three keys regardless of training loss. mean-of-K ADE is
    the trial comparison metric (matches what --loss mean_ade is optimizing);
    the others are tracked for context.
    """
    model.eval()
    sum_min_ade = 0.0
    sum_mean_ade = 0.0
    sum_min_fde = 0.0
    n_total = 0

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
        y_exp = y[:, :, None, :, :]                               # [B, N, 1, T_f, 2]

        per_step = torch.norm(y_pred - y_exp, dim=-1)             # [B, N, K, T_f]
        ade_perK = per_step.mean(dim=3)                           # [B, N, K]
        min_ade = ade_perK.min(dim=2)[0]                          # [B, N]
        mean_ade = ade_perK.mean(dim=2)                           # [B, N]
        min_fde = per_step[..., -1].min(dim=2)[0]                 # [B, N]

        sum_min_ade  += min_ade.sum().item()
        sum_mean_ade += mean_ade.sum().item()
        sum_min_fde  += min_fde.sum().item()
        n_total      += B * N

    return {
        'minADE':  sum_min_ade  / max(n_total, 1),
        'meanADE': sum_mean_ade / max(n_total, 1),
        'minFDE':  sum_min_fde  / max(n_total, 1),
    }


# ------------------------ Per-trial training ------------------

def train_one_trial(opts, train_loader, val_loader, train_sampler,
                    device, num_epochs, wandb_prefix=''):
    """Train for `num_epochs`. No pruning, no save inside (caller decides).

    mu/sigma are already baked into the loaders (used to z-score at dataset
    construction), so this function doesn't need them. The caller checkpoints
    the best trial and is responsible for stashing mu/sigma there.

    Returns:
        last_metrics: dict from eval_val_metrics on the final epoch
        last_state_dict: model.state_dict() on CPU (None if training NaN'd)
    """
    model = build_model(opts, device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=opts.lr,
        weight_decay=opts.get('weight_decay', 1e-12),
    )
    scheduler = _make_scheduler(opts, optimizer)

    last_metrics = None
    last_state_dict = None

    try:
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            # aug_offset=None (default) -> no augmentation, per sweep spec.
            train_loss = train_one_epoch(
                epoch, model, optimizer, train_loader, opts, device,
            )
            val_metrics = eval_val_metrics(model, val_loader, opts, device)

            if scheduler is not None:
                scheduler.step()

            wandb.log({
                f'{wandb_prefix}epoch': epoch,
                f'{wandb_prefix}lr': optimizer.param_groups[0]['lr'],
                f'{wandb_prefix}train/loss': train_loss,
                f'{wandb_prefix}val/meanADE': val_metrics['meanADE'],
                f'{wandb_prefix}val/minADE':  val_metrics['minADE'],
                f'{wandb_prefix}val/minFDE':  val_metrics['minFDE'],
            })

            # Early bail-out on numerical blowups. We let Optuna treat the
            # trial as failed (NaN as the objective) rather than save garbage.
            if not math.isfinite(train_loss) or not math.isfinite(val_metrics['meanADE']):
                print(f'  epoch {epoch + 1}/{num_epochs} : non-finite loss/metric — aborting trial')
                return None, None

            last_metrics = val_metrics
            print(
                f'  epoch {epoch + 1:3d}/{num_epochs} | '
                f'train_loss={train_loss:.4f} | '
                f'val_meanADE={val_metrics["meanADE"]:.4f} | '
                f'val_minADE={val_metrics["minADE"]:.4f} | '
                f'val_minFDE={val_metrics["minFDE"]:.4f}'
            )

        # Move state dict off the GPU so we don't hold model memory across
        # trials when the caller decides to checkpoint a new best.
        last_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    finally:
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return last_metrics, last_state_dict


# ------------------------ Optuna entry ------------------------

def main():
    base = parse_cli()
    setup_seed(base.seed)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(base.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] device:', device)

    base_opts = load_config(base.config)
    print('[INFO] base opts:', base_opts)

    # One-time data stats reused for normalization across all trials.
    train_files, _ = load_split_files(base.split_path)
    print(f'[INFO] computing stats from {len(train_files)} train files...')
    mu, sigma = compute_xy_stats(train_files)
    print(f'[INFO] mu={mu.tolist()} sigma={sigma.tolist()}')

    # If --final_model_name is absolute, os.path.join drops the './checkpoints'
    # prefix and the path is used as-is. Either way, make the destination
    # directory before training so we don't lose 100 epochs to a missing dir.
    final_ckpt_path = os.path.join('./checkpoints', f'{base.final_model_name}.ckpt')
    os.makedirs(os.path.dirname(os.path.abspath(final_ckpt_path)), exist_ok=True)
    print(f'[INFO] final-retrain checkpoint will be written to: {final_ckpt_path}')

    def objective(trial):
        opts = apply_search_params(base_opts, suggest_search_params(trial))
        print(f'\n----- TRIAL {trial.number} -----')
        for k in ('lr', 'weight_decay', 'dropout', 'num_layers',
                  'model_dim', 'hidden_dim', 'decoder_hidden_dim'):
            print(f'  {k:<22s} = {opts[k]}')
        print(f'  loss={opts.loss}  use_hoops={opts.use_hoops}  '
              f'aug_court_mirror={opts.aug_court_mirror}')

        train_loader, val_loader, train_sampler = build_loaders(
            opts, mu, sigma, base.split_path, base.num_workers, base.seed,
        )

        run_name = (
            f'trial_{trial.number}_lr{opts.lr:.1e}_wd{opts.weight_decay:.1e}_'
            f'md{opts.model_dim}_hd{opts.hidden_dim}_'
            f'dhd{opts.decoder_hidden_dim}_nl{opts.num_layers}_do{opts.dropout}'
        )
        with wandb.init(
            project=base.wandb_project, name=run_name,
            mode=base.wandb_mode,
            config={'trial_number': trial.number, **dict(opts)},
            reinit=True,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        ):
            last_metrics, _ = train_one_trial(
                opts,
                train_loader, val_loader, train_sampler, device,
                num_epochs=base.trial_epochs,
            )
            if last_metrics is None:
                wandb.log({'last_val/meanADE': float('nan')})
                raise optuna.TrialPruned('non-finite loss / metric')

            last_mean_ade = last_metrics['meanADE']
            wandb.log({
                'last_val/meanADE': last_mean_ade,
                'last_val/minADE':  last_metrics['minADE'],
                'last_val/minFDE':  last_metrics['minFDE'],
            })

        return last_mean_ade

    def _format_param(k, v):
        v_str = f'{v:.3e}' if isinstance(v, float) and v < 1e-2 else f'{v}'
        return f'{k:<22s} = {v_str}'

    def progress_cb(study, trial):
        """Live-print a new-best banner whenever this trial improved the study."""
        # study.best_trial only exists once at least one trial completed.
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_trial.number != trial.number:
            return
        print(f'\n[NEW BEST] after trial {trial.number}: '
              f'val meanADE = {study.best_value:.4f}')
        print('  best hyperparams so far:')
        for k, v in study.best_params.items():
            print(f'      {_format_param(k, v)}')

    # No pruner: with last-epoch comparison every trial must run to completion.
    study = optuna.create_study(
        direction='minimize',
        study_name=base.study_name,
        storage=base.storage,
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
    )
    print(f'\n========== SEARCH ({base.n_trials} trials x '
          f'{base.trial_epochs} epochs) ==========')
    print('  loss            = mean_ade  (forced)')
    print('  use_hoops       = False     (forced)')
    print('  aug_court_mirror= False     (forced)')
    print('  selection       = last-epoch val meanADE (minimize)')
    study.optimize(objective, n_trials=base.n_trials, callbacks=[progress_cb])

    # ---- Search-complete summary ----
    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_failed = len([t for t in study.trials
                    if t.state != optuna.trial.TrialState.COMPLETE])
    print('\n========== SEARCH COMPLETE ==========')
    print(f'  selection metric : last-epoch val meanADE (minimize)')
    print(f'  trials run       : {len(study.trials)}  '
          f'(completed={n_complete}, failed/pruned={n_failed})')
    if n_complete == 0:
        print('  no successful trial — skipping final retrain')
        return
    print(f'  best trial #     : {study.best_trial.number}')
    print(f'  best meanADE     : {study.best_value:.4f}')
    print('  best hyperparams :')
    for k, v in study.best_params.items():
        print(f'      {_format_param(k, v)}')

    # ---- FINAL RETRAIN with the winning HPs ----
    print(f'\n========== FINAL RETRAIN ({base.final_epochs} epochs) ==========')
    final_opts = apply_search_params(base_opts, study.best_params)
    train_loader, val_loader, train_sampler = build_loaders(
        final_opts, mu, sigma, base.split_path, base.num_workers, base.seed,
    )

    # Re-seed: the search above consumed the global torch RNG, so the model
    # init drifts from a clean main_nba_pt.py run unless we reset here.
    setup_seed(base.seed)
    print(f'[INFO] re-seeded for final retrain (seed={base.seed})')

    with wandb.init(
        project=base.wandb_project, name=f'{base.study_name}_final_retrain',
        mode=base.wandb_mode,
        config={**dict(final_opts), 'best_search_params': study.best_params},
        reinit=True,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    ):
        final_metrics, final_state_dict = train_one_trial(
            final_opts,
            train_loader, val_loader, train_sampler, device,
            num_epochs=base.final_epochs,
            wandb_prefix='final_',
        )
        if final_metrics is None:
            print('[ERROR] final retrain diverged (non-finite loss). '
                  'No checkpoint written.')
            return
        wandb.log({
            'final_last_val/meanADE': final_metrics['meanADE'],
            'final_last_val/minADE':  final_metrics['minADE'],
            'final_last_val/minFDE':  final_metrics['minFDE'],
        })

    # Same on-disk format main_nba_pt.py writes — eval.py + submit_nba_pt.py
    # consume this checkpoint directly.
    torch.save(
        {
            'epoch': base.final_epochs - 1,
            'state_dict': final_state_dict,
            'opts': dict(final_opts),
            'mu': mu,
            'sigma': sigma,
            'val_meanADE': final_metrics['meanADE'],
            'val_minADE':  final_metrics['minADE'],
            'val_minFDE':  final_metrics['minFDE'],
            'best_search_params': dict(study.best_params),
        },
        final_ckpt_path,
    )

    # ---- Final summary ----
    print('\n========== FINAL SUMMARY ==========')
    print(f'  selection metric : last-epoch val meanADE (minimize)')
    print(f'  search budget    : {base.n_trials} trials x {base.trial_epochs} epochs')
    print(f'  final retrain    : {base.final_epochs} epochs (seed={base.seed})')
    print('  winning hyperparams (from search):')
    for k, v in study.best_params.items():
        print(f'      {_format_param(k, v)}')
    print('  final retrain val metrics (last epoch, normalized units):')
    print(f'      val meanADE = {final_metrics["meanADE"]:.4f}')
    print(f'      val minADE  = {final_metrics["minADE"]:.4f}')
    print(f'      val minFDE  = {final_metrics["minFDE"]:.4f}')
    print(f'  saved checkpoint : {final_ckpt_path}')
    print(
        '\n[INFO] Evaluate this checkpoint in feet with eval.py, or generate '
        'a submission with submit_nba_pt.py — both auto-detect the loss / '
        'flags from the checkpoint.'
    )


if __name__ == '__main__':
    main()
