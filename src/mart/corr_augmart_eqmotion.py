"""Cross-architecture error-correlation + ensemble analysis: aug-MART vs EqMotion.

Motivation: the earlier ρ≈0.93 finding (which killed cross-arch averaging) was
measured on the OLD under-trained MART (val 3.79). aug-MART (val 3.11 / Kaggle
3.01) is now the STRONG model. If its errors are even partly decorrelated from
the EqMotion ensemble's, an aug-MART-anchored weighted average could beat 3.11.

Both caches are on the SAME 3078 deterministic val windows (WindowEvalSampler,
8/seq) in canonical agent order [TeamA*5, TeamB*5, Ball]; we verify target
alignment in feet before trusting any comparison.

Inputs (val caches):
  - cache/eqm_residual/val.pt    : eqm_pred_ft [M,11,12,2] (FEET), target [norm],
                                   mart_mu/mart_sigma (aniso, for target denorm)
  - <augmart_cache>/val.pt       : k_preds [M,11,20,12,2] (z-score, iso), target
                                   [norm, iso], mu/sigma (iso)

Usage:
    python corr_augmart_eqmotion.py \\
        --augmart_cache ../../cache/mart_aug_5k \\
        --eqm_cache ../../cache/eqm_residual
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch


def per_window_entity_mse(pred_ft, tgt_ft):
    """[M,11,12,2] feet -> [M,11] mean-squared-error over (T, xy)."""
    return ((pred_ft - tgt_ft) ** 2).mean(dim=(2, 3))


def summarize(name, pred_ft, tgt_ft, ball_idx=10):
    e = per_window_entity_mse(pred_ft, tgt_ft)  # [M,11]
    total = e.mean().item()
    ball = e[:, ball_idx].mean().item()
    players = e[:, [i for i in range(11) if i != ball_idx]].mean().item()
    print(f'  {name:24s} total={total:.3f}  ball={ball:.3f}  players={players:.3f}')
    return e, total, ball, players


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--augmart_cache', required=True)
    p.add_argument('--eqm_cache', default='../../cache/eqm_residual')
    args = p.parse_args()

    eqm = torch.load(os.path.join(args.eqm_cache, 'val.pt'), weights_only=False)
    aug = torch.load(os.path.join(args.augmart_cache, 'val.pt'), weights_only=False)

    # --- Denormalize everything to FEET in each cache's own frame ---
    # EqMotion preds already in feet; its target is normalized with mart_mu/sigma.
    eqm_mu = eqm['mart_mu'].view(1, 1, 1, 2)
    eqm_sig = eqm['mart_sigma'].view(1, 1, 1, 2)
    tgt_eqm_ft = eqm['target'] * eqm_sig + eqm_mu
    eqm_ft = eqm['eqm_pred_ft']

    # aug-MART: k_preds + target are z-scored with the iso mu/sigma.
    aug_mu = aug['mu'].view(1, 1, 1, 2)
    aug_sig = aug['sigma'].view(1, 1, 1, 2)
    tgt_aug_ft = aug['target'] * aug_sig + aug_mu
    augmart_ft = (aug['k_preds'].mean(dim=2)) * aug_sig + aug_mu  # mean-of-K, feet

    # --- Alignment sanity: the two caches must be the SAME windows ---
    same = torch.allclose(tgt_eqm_ft, tgt_aug_ft, atol=1e-2)
    max_diff = (tgt_eqm_ft - tgt_aug_ft).abs().max().item()
    print(f'[ALIGN] targets match in feet: {same} (max abs diff {max_diff:.4f} ft)')
    if not same:
        print('[ALIGN] !! windows are NOT aligned — abort, correlation is meaningless.')
        sys.exit(1)
    tgt_ft = tgt_eqm_ft  # canonical

    print(f'[INFO] M={tgt_ft.shape[0]} val windows, 11 entities, ball=idx10\n')

    print('=== Single-model val/mse_ft (sanity vs known numbers) ===')
    e_eqm, t_eqm, b_eqm, p_eqm = summarize('EqMotion ensemble', eqm_ft, tgt_ft)
    e_aug, t_aug, b_aug, p_aug = summarize('aug-MART 5k (mean-K)', augmart_ft, tgt_ft)
    print('  (expect EqMotion ~3.29, aug-MART ~3.11)\n')

    # --- Error correlation across windows (per slice) ---
    print('=== Per-entity error correlation ρ (across windows) ===')
    def corr(a, b):
        a = a.flatten().numpy(); b = b.flatten().numpy()
        return float(np.corrcoef(a, b)[0, 1])
    players_cols = [i for i in range(11) if i != 10]
    rho_all = corr(e_aug, e_eqm)
    rho_ball = corr(e_aug[:, 10], e_eqm[:, 10])
    rho_pl = corr(e_aug[:, players_cols], e_eqm[:, players_cols])
    print(f'  all-11 ρ = {rho_all:.3f}   ball ρ = {rho_ball:.3f}   players ρ = {rho_pl:.3f}')
    print('  (old under-trained MART had ρ≈0.93 across all slices)\n')

    # --- Weighted ensemble sweep: w*augMART + (1-w)*EqMotion, in feet ---
    print('=== Weighted ensemble: w·augMART + (1-w)·EqMotion ===')
    print(f'  {"w_aug":>6} {"total":>8} {"ball":>8} {"players":>8}')
    best = (None, 1e9)
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * augmart_ft + (1 - w) * eqm_ft
        eb = per_window_entity_mse(blend, tgt_ft)
        tot = eb.mean().item()
        ball = eb[:, 10].mean().item()
        pl = eb[:, players_cols].mean().item()
        mark = ''
        if abs(w - 1.0) < 1e-9: mark = '  <- aug-MART alone'
        if abs(w) < 1e-9: mark = '  <- EqMotion alone'
        print(f'  {w:6.2f} {tot:8.3f} {ball:8.3f} {pl:8.3f}{mark}')
        if tot < best[1]:
            best = (w, tot)
    print(f'\n  BEST blend: w_aug={best[0]:.2f} -> total={best[1]:.4f}')
    print(f'  vs aug-MART alone {t_aug:.4f}  | gain {t_aug - best[1]:+.4f}')

    # --- Closed-form optimal per-entity-class weight (analytic check) ---
    # For each slice, optimal w minimizing ||w*a+(1-w)*e - t||^2 over residuals:
    # let da = augMART - target, de = EqMotion - target; w* = <de-da, de> / ||de-da||^2...
    # simpler: w* = <de, de-da_resid>; use vectorized lstsq on residual vectors.
    def opt_w(cols):
        a = (augmart_ft[:, cols] - tgt_ft[:, cols]).flatten().numpy()
        e = (eqm_ft[:, cols] - tgt_ft[:, cols]).flatten().numpy()
        # minimize var of w*a+(1-w)*e ; w* = <e-a, e>/<e-a,e-a>
        d = e - a
        denom = float((d * d).sum())
        if denom == 0: return float('nan')
        return float((d * e).sum() / denom)
    print('\n=== Closed-form optimal aug-MART weight (per slice) ===')
    print(f'  all-11  w* = {opt_w(list(range(11))):.3f}')
    print(f'  ball    w* = {opt_w([10]):.3f}')
    print(f'  players w* = {opt_w(players_cols):.3f}')
    print('  (w*≈1 => EqMotion adds nothing; w*<1 => ensembling helps)')


if __name__ == '__main__':
    main()
