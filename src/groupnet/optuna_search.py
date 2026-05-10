"""Optuna hyperparameter search for GroupNetWithID, then retrain the winner.

Two-stage workflow:
    1. SEARCH: run N trials, each training a short version of the model
       (--trial_epochs) and reporting val ADE. Bad trials get pruned early.
    2. FINAL RETRAIN: take the winning config and retrain from scratch for
       --final_epochs, saving checkpoints every 10 epochs + a final checkpoint
       under --final_save_dir. Earlier trial models are NOT saved.

Search space (the knobs that affect ADE most):
    - hidden_dim     : 64, 128, 256
    - lr             : 5e-5 .. 5e-4, log-uniform
    - decay_step     : 10, 20, 30
    - embed_dim      : 4, 8, 16
    - num_decompose  : 2, 3

Notes on what is NOT searched:
    - sample_k (training-time best-of-K) stays at 20 (paper default; affects
      only the diverse loss).
    - K_eval (val ADE samples) is fixed via --K_eval so trials are comparable.
      Sweep K_eval afterward with eval_k_sweep.py — it's free, no retrain.

Example:
    python optuna_search.py \
        --split_path ../network_ml_project/splits/fold0.json \
        --n_trials 8 \
        --trial_epochs 20 \
        --final_epochs 100 \
        --K_eval 50 \
        --gpu 0
"""

import argparse
import os
import sys
from types import SimpleNamespace

sys.path.append(os.getcwd())

# wandb auth (replace before running on cluster)
os.environ.setdefault('WANDB_API_KEY', 'put your_api_key_here')

import optuna
import torch
import wandb
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data.dataloader_nba_pt import (
    GroupNetNBAPTDataset,
    WindowSampler,
    compute_xy_stats,
    groupnet_collate,
    load_split_files,
)
from model.GroupNet_nba_id import GroupNetWithID
from submit_nba_pt import inference_chunked


# ----------------------------- CLI -----------------------------

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument('--split_path', required=True,
                   help='Path to network_ml_project splits/<name>.json')
    # Search budget
    p.add_argument('--n_trials', type=int, default=8)
    p.add_argument('--trial_epochs', type=int, default=20,
                   help='epochs per Optuna trial (kept short for speed)')
    p.add_argument('--final_epochs', type=int, default=100,
                   help='epochs for the final retrain of the winning config')
    # Eval setup
    p.add_argument('--K_eval', type=int, default=50,
                   help='samples for val ADE; fixed across trials for fairness')
    p.add_argument('--sample_k_chunk', type=int, default=50)
    # Data / runtime
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--past_length', type=int, default=8)
    p.add_argument('--future_length', type=int, default=12)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    # Outputs
    p.add_argument('--final_save_dir', default='saved_models/optuna_best',
                   help='where the winning config\'s checkpoints are saved')
    p.add_argument('--study_name', default='groupnet_nba_optuna')
    p.add_argument('--storage', default=None,
                   help='optional sqlite path for study persistence, '
                        'e.g. sqlite:///optuna.db (lets you resume after a crash)')
    p.add_argument('--wandb_project', default='groupnet_nba_optuna')
    return p.parse_args()


# ------------------------ Config plumbing ----------------------

def base_fixed_cfg(base):
    """Return a SimpleNamespace with all the cfg fields the model needs."""
    cfg = SimpleNamespace(**vars(base))
    # Fixed model knobs (paper defaults)
    cfg.zdim = 32
    cfg.hyper_scales = [5, 11]
    cfg.min_clip = 2.0
    cfg.ztype = 'gaussian'
    cfg.sample_k = 20            # training-time best-of-K
    cfg.learn_prior = False
    cfg.decay_gamma = 0.5
    cfg.traj_scale = 1
    cfg.iternum_print = 200
    cfg.epoch_continue = 0
    cfg.model_save_epoch = 10    # final retrain saves every 10 epochs
    return cfg


def apply_search_params(cfg, params):
    cfg.hidden_dim = params['hidden_dim']
    cfg.lr = params['lr']
    cfg.decay_step = params['decay_step']
    cfg.embed_dim = params['embed_dim']
    cfg.num_decompose = params['num_decompose']
    return cfg


def suggest_search_params(trial):
    return {
        'hidden_dim':    trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'lr':            trial.suggest_float('lr', 5e-5, 5e-4, log=True),
        'decay_step':    trial.suggest_categorical('decay_step', [10, 20, 30]),
        'embed_dim':     trial.suggest_categorical('embed_dim', [4, 8, 16]),
        'num_decompose': trial.suggest_categorical('num_decompose', [2, 3]),
    }


# ------------------------ Data + eval --------------------------

def build_loaders(cfg, mu, sigma):
    train_files, val_files = load_split_files(cfg.split_path)
    train_set = GroupNetNBAPTDataset(train_files, mu, sigma,
                                     cfg.past_length, cfg.future_length)
    val_set = GroupNetNBAPTDataset(val_files, mu, sigma,
                                   cfg.past_length, cfg.future_length)
    train_sampler = WindowSampler(cfg.batch_size, train_set.max_start,
                                  seed=cfg.seed, shuffle=True)
    val_sampler = WindowSampler(cfg.batch_size, val_set.max_start,
                                seed=cfg.seed, shuffle=False)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, collate_fn=groupnet_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.num_workers, collate_fn=groupnet_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler


def compute_val_ade(model, val_loader, mu, sigma, future_length,
                    K_eval, sample_k_chunk):
    """Average displacement error on val set, in court units."""
    model.eval()
    sigma_b = sigma.view(1, 1, 1, 2)
    mu_b = mu.view(1, 1, 1, 2)
    err_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for data in val_loader:
            pred = inference_chunked(model, data, total_k=K_eval,
                                     chunk_k=sample_k_chunk)
            B = data['past_traj'].shape[0]
            N = data['past_traj'].shape[1]
            pred = pred.view(K_eval, B, N, future_length, 2).cpu()
            pred_mean = pred.mean(dim=0)
            pred_denorm = pred_mean * sigma_b + mu_b
            target = data['future_traj'] * sigma_b + mu_b
            err = torch.norm(pred_denorm - target, dim=-1)  # [B, N, T_f]
            err_sum += err.mean(dim=-1).sum().item()
            n_total += err.shape[0] * err.shape[1]
    return err_sum / max(n_total, 1)


# ------------------------ Train loop --------------------------

def train_run(cfg, mu, sigma, train_loader, val_loader, train_sampler, device,
              *, save_checkpoints=False, save_dir=None, trial=None,
              wandb_prefix=''):
    """Train for cfg.num_epochs. Returns the best val ADE seen during the run.

    If save_checkpoints, dumps a .p file every cfg.model_save_epoch epochs and
    one final .p at the end.
    If trial is provided, intermediate val ADEs are reported for Optuna pruning.
    """
    if save_checkpoints:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({'mu': mu, 'sigma': sigma},
                   os.path.join(save_dir, 'norm_stats.pt'))

    model = GroupNetWithID(cfg, device, embed_dim=cfg.embed_dim)
    model.set_device(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=cfg.decay_step,
                                    gamma=cfg.decay_gamma)

    best_val_ade = float('inf')
    for epoch in range(cfg.num_epochs):
        # ---- train ----
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for data in train_loader:
            total_loss, *_ = model(data)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
            n_batches += 1
        scheduler.step()
        model.step_annealer()

        train_total = train_loss_sum / max(n_batches, 1)

        # ---- val ----
        val_ade = compute_val_ade(model, val_loader, mu, sigma,
                                  cfg.future_length, cfg.K_eval,
                                  cfg.sample_k_chunk)
        if val_ade < best_val_ade:
            best_val_ade = val_ade

        print(f'  epoch {epoch + 1:3d}/{cfg.num_epochs} | '
              f'train_total={train_total:.4f} | val_ADE={val_ade:.4f}')
        wandb.log({
            f'{wandb_prefix}train/total': train_total,
            f'{wandb_prefix}val/ADE': val_ade,
            f'{wandb_prefix}val/best_ADE': best_val_ade,
            'epoch': epoch,
        })

        # ---- pruning ----
        if trial is not None:
            trial.report(val_ade, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # ---- checkpoint ----
        if save_checkpoints:
            is_save_epoch = (epoch + 1) % cfg.model_save_epoch == 0
            is_final = (epoch + 1) == cfg.num_epochs
            if is_save_epoch or is_final:
                ckpt = {
                    'model_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'model_cfg': cfg,
                    'mu': mu,
                    'sigma': sigma,
                    'val_ade': val_ade,
                }
                save_path = os.path.join(save_dir, f'{epoch + 1}.p')
                torch.save(ckpt, save_path)
                print(f'    saved {save_path}')

    # free GPU between trials
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_ade


# ------------------------ Optuna entry ------------------------

def main():
    base = parse_cli()
    device = (torch.device('cuda', base.gpu)
              if torch.cuda.is_available() else torch.device('cpu'))
    print('device:', device)

    # one-time data stats
    train_files, _ = load_split_files(base.split_path)
    print(f'computing stats from {len(train_files)} train files...')
    mu, sigma = compute_xy_stats(train_files)
    print(f'mu={mu.tolist()} sigma={sigma.tolist()}')

    # ---- Optuna search ----
    def objective(trial):
        cfg = base_fixed_cfg(base)
        cfg = apply_search_params(cfg, suggest_search_params(trial))
        cfg.num_epochs = base.trial_epochs

        train_loader, val_loader, train_sampler = build_loaders(cfg, mu, sigma)

        run_name = (f'trial_{trial.number}_hd{cfg.hidden_dim}'
                    f'_lr{cfg.lr:.1e}_ds{cfg.decay_step}'
                    f'_ed{cfg.embed_dim}_nd{cfg.num_decompose}')
        with wandb.init(project=base.wandb_project, name=run_name,
                        config=vars(cfg), reinit=True,
                        settings=wandb.Settings(_disable_stats=True,
                                                _disable_meta=True)):
            best_ade = train_run(
                cfg, mu, sigma, train_loader, val_loader, train_sampler,
                device, save_checkpoints=False, trial=trial,
            )
            wandb.log({'best_val_ade_in_trial': best_ade})
        return best_ade

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    study = optuna.create_study(
        direction='minimize',
        study_name=base.study_name,
        storage=base.storage,
        load_if_exists=True,
        pruner=pruner,
    )
    print(f'\n========== SEARCH ({base.n_trials} trials) ==========')
    study.optimize(objective, n_trials=base.n_trials)

    print('\n========== SEARCH COMPLETE ==========')
    print(f'Best trial    : {study.best_trial.number}')
    print(f'Best val ADE  : {study.best_value:.4f}')
    print(f'Best params   : {study.best_params}')

    # ---- Final retrain ----
    print(f'\n========== FINAL RETRAIN ({base.final_epochs} epochs) ==========')
    final_cfg = base_fixed_cfg(base)
    final_cfg = apply_search_params(final_cfg, study.best_params)
    final_cfg.num_epochs = base.final_epochs

    train_loader, val_loader, train_sampler = build_loaders(final_cfg, mu, sigma)

    with wandb.init(project=base.wandb_project, name='best_retrain',
                    config=vars(final_cfg), reinit=True,
                    settings=wandb.Settings(_disable_stats=True,
                                            _disable_meta=True)):
        final_best_ade = train_run(
            final_cfg, mu, sigma, train_loader, val_loader, train_sampler,
            device, save_checkpoints=True, save_dir=base.final_save_dir,
            wandb_prefix='final_',
        )
        wandb.log({'final_best_val_ade': final_best_ade})

    print(f'\nFinal best val ADE: {final_best_ade:.4f}')
    print(f'Checkpoints saved in: {base.final_save_dir}')


if __name__ == '__main__':
    main()
