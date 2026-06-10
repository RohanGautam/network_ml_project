"""Generate the report figures.

Fig 2 (training curves) is built from W&B validation logs:
  src/report/wandb_val_mse_ft.csv               (run, epoch, val_mse_ft)
  src/report/mart curriculum training val_mse log.csv  (curriculum run)
  -> run `uv run python src/report/fetch_wandb_curves.py` first to (re)download.

Figs 3 and 4 are built from cached validation predictions (no model inference):
  cache/mart_aug_5k/val.pt   (aug-MART: past, target, k_preds; z-scored iso frame)
  cache/eqm_residual/val.pt  (EqMotion ensemble: eqm_pred_ft in feet; target)

Fig 2: side-by-side val/mse_ft training curves. Left: the three MART variants
       that differ only in training loss (min-ADE / mean-MSE / curriculum) ->
       mean-MSE overfits, curriculum wins. Right: best MART vs EqMotion.
Fig 3: per-window BALL error of EqMotion vs aug-MART -> the rho~0.93 scatter that
       visually argues "same errors on the same samples => task floor."
Fig 4: one play on the court -- players predicted tightly, the ball's K=20 modes
       fanning out to plausible pass targets -> ball multimodality made tangible.

Usage:  uv run python src/report/make_report_figs.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
COURT_IMAGE = ROOT / "src" / "img" / "basketball_court.png"
OUT = Path(__file__).resolve().parent
BALL = 10  # canonical agent order: [TeamA*5, TeamB*5, Ball]


def to_display(xy):
    """Centered feet (origin = court center) -> court display coords [0,95]x[0,50]."""
    out = np.asarray(xy, dtype=float).copy()
    out[..., 0] += 47.5
    out[..., 1] += 25.0
    return out


def setup_court(ax, title=""):
    court = plt.imread(str(COURT_IMAGE))
    ax.imshow(court, extent=[0, 95, 0, 50], aspect="auto", zorder=0, alpha=0.5)
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 50)
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    if title:
        ax.set_title(title)


def per_window_entity_mse(pred_ft, tgt_ft):
    """[M,11,12,2] feet -> [M,11] MSE over (T, xy)."""
    return ((pred_ft - tgt_ft) ** 2).mean(axis=(2, 3))


# ---------------------------------------------------------------------------
# Figure 2 -- validation training curves (loss comparison + MART vs EqMotion)
# ---------------------------------------------------------------------------
def make_training_curves():
    """Two side-by-side val/mse_ft panels, built from the W&B logs."""
    wb = pd.read_csv(OUT / "wandb_val_mse_ft.csv")

    def curve(run, smooth=11):
        g = wb[wb.run == run].sort_values("epoch")
        x = g.epoch.to_numpy(dtype=float)
        y = g.val_mse_ft.to_numpy(dtype=float)
        ys = pd.Series(y).rolling(smooth, min_periods=1, center=True).mean().to_numpy()
        return x, y, ys

    # The curriculum run was not retained in W&B; use the CSV the user exported.
    cur = pd.read_csv(OUT / "mart curriculum training val_mse log.csv")
    cur_x = cur["Step"].to_numpy(dtype=float)
    cur_y = cur.iloc[:, 1].to_numpy(dtype=float)
    cur_ys = pd.Series(cur_y).rolling(11, min_periods=1, center=True).mean().to_numpy()

    SWITCH = 1000          # curriculum loss switch (min-ADE -> mean-MSE) + LR drop
    YMIN, YMAX = 2.8, 5.0  # zoom to the plateau where the variants separate

    C_MINADE = "#1E88E5"
    C_MEANMSE = "#9E9E9E"
    C_CURR = "#E53935"
    C_EQM = "#00897B"

    def plot_run(ax, x, y_raw, y_smooth, color, label):
        ax.plot(x, y_raw, color=color, lw=0.8, alpha=0.25)
        ax.plot(x, y_smooth, color=color, lw=1.6, label=label)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)

    # --- Left: MART, holding everything fixed except the training loss ---
    mx, my, mys = curve("mart_meanmse_iso_hoops_5k")
    ax_, ay, ays = curve("mart_aug_iso_hoops_5k")
    plot_run(axL, mx, my, mys, C_MEANMSE, f"mean-MSE  (best {my.min():.2f})")
    plot_run(axL, ax_, ay, ays, C_MINADE, f"min-ADE  (best {ay.min():.2f})")
    plot_run(axL, cur_x, cur_y, cur_ys, C_CURR, f"curriculum  (best {cur_y.min():.2f})")
    axL.axvline(SWITCH, ls=":", color="k", lw=1.0, alpha=0.6)
    axL.annotate("loss switch", xy=(SWITCH + 80, 4.93), ha="left", va="top",
                 fontsize=7, color="0.3")
    axL.set_title("(a) MART: effect of training loss")
    axL.legend(frameon=False, fontsize=8, loc="upper right")

    # --- Right: the best MART vs the EqMotion baseline ---
    qx, qy, qys = curve("eqm_iso_hoops_5k")
    plot_run(axR, qx, qy, qys, C_EQM, f"EqMotion  (best {qy.min():.2f})")
    plot_run(axR, cur_x, cur_y, cur_ys, C_CURR, f"MART curriculum  (best {cur_y.min():.2f})")
    axR.set_title("(b) Best MART vs EqMotion")
    axR.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in (axL, axR):
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlim(0, 5000)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axL.set_ylabel("validation MSE (ft$^2$)")

    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.png", dpi=200)
    fig.savefig(OUT / "fig_training_curves.pdf")
    plt.close(fig)
    print(f"[fig2] saved {OUT/'fig_training_curves.png'}  "
          f"(min-ADE {ay.min():.2f}, mean-MSE {my.min():.2f}->{my[-1]:.2f}, "
          f"curriculum {cur_y.min():.2f}, EqMotion {qy.min():.2f})")


make_training_curves()


# ---------------------------------------------------------------------------
# Load + denormalize both caches to feet on the SAME windows
# ---------------------------------------------------------------------------
aug = torch.load(ROOT / "cache/mart_aug_5k/val.pt", weights_only=False)
eqm = torch.load(ROOT / "cache/eqm_residual/val.pt", weights_only=False)

a_mu, a_sig = aug["mu"].view(1, 1, 1, 2), aug["sigma"].view(1, 1, 1, 2)
tgt_ft = (aug["target"] * a_sig + a_mu).numpy()                       # [M,11,12,2]
augmart_ft = (aug["k_preds"].mean(dim=2) * a_sig + a_mu).numpy()      # mean-of-K
past_ft = (aug["past"] * a_sig + a_mu).numpy()                        # [M,11,8,2]
kpreds_ft = (aug["k_preds"] * a_sig.unsqueeze(2) + a_mu.unsqueeze(2)).numpy()  # [M,11,20,12,2]

e_mu, e_sig = eqm["mart_mu"].view(1, 1, 1, 2), eqm["mart_sigma"].view(1, 1, 1, 2)
tgt_eqm_ft = (eqm["target"] * e_sig + e_mu).numpy()
eqm_ft = eqm["eqm_pred_ft"].numpy()

max_diff = np.abs(tgt_ft - tgt_eqm_ft).max()
assert max_diff < 1e-2, f"caches not aligned ({max_diff:.4f} ft) -- aborting"
print(f"[align] targets match in feet (max diff {max_diff:.4g}); M={tgt_ft.shape[0]} windows")

e_aug = per_window_entity_mse(augmart_ft, tgt_ft)   # [M,11]
e_eqm = per_window_entity_mse(eqm_ft, tgt_ft)

# ---------------------------------------------------------------------------
# Figure 3 -- cross-architecture ball-error scatter
# ---------------------------------------------------------------------------
xb, yb = e_eqm[:, BALL], e_aug[:, BALL]
rho = float(np.corrcoef(xb, yb)[0, 1])
print(f"[fig3] ball rho = {rho:.3f}  (EqMotion mean {xb.mean():.2f}, aug-MART {yb.mean():.2f})")

fig, ax = plt.subplots(figsize=(4.2, 4.0))
lim = np.percentile(np.concatenate([xb, yb]), 99)  # clip the long tail for readability
ax.scatter(xb, yb, s=6, alpha=0.25, color="#1E88E5", edgecolors="none")
ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="y = x")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("EqMotion ensemble  ball MSE (ft$^2$)")
ax.set_ylabel("aug-MART  ball MSE (ft$^2$)")
ax.set_title(f"Per-play ball error  ($\\rho = {rho:.2f}$)")
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3.png", dpi=200)
plt.close(fig)
print(f"[fig3] saved {OUT/'fig3.png'}")

# ---------------------------------------------------------------------------
# Figure 4 -- one play: tight players, fanning ball modes
# ---------------------------------------------------------------------------
# Pick a legible, clearly-multimodal example: large spread among the K ball
# endpoints (genuine pass ambiguity) but a ball that actually travels.
end_k = kpreds_ft[:, BALL, :, -1, :]                       # [M,20,2] final ball pos per head
spread = end_k.std(axis=1).sum(axis=1)                     # [M] endpoint dispersion
travel = np.linalg.norm(tgt_ft[:, BALL, -1] - past_ft[:, BALL, -1], axis=1)  # ball displacement
cand = np.where(travel > 12)[0]                            # ball moves a meaningful distance
idx = cand[np.argmax(spread[cand])]
print(f"[fig4] window {idx}: ball travel {travel[idx]:.1f} ft, K-endpoint spread {spread[idx]:.1f}")

fig, ax = plt.subplots(figsize=(7.8, 4.4))
setup_court(ax)
# Convention: solid line = past context (leads INTO the dot), dot = position at
# the prediction instant t=C, dashed/fan = future (leads OUT of the dot).
# Players first, de-emphasized, so the ball reads as the focus.
for n in range(11):
    if n == BALL:
        continue
    col = "#E53935" if n < 5 else "#1E88E5"
    p = to_display(past_ft[idx, n]); g = to_display(tgt_ft[idx, n])
    ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.3, alpha=0.55, zorder=2)
    ax.plot([p[-1, 0], g[0, 0]], [p[-1, 1], g[0, 1]], "-", color=col, lw=1.3, alpha=0.55, zorder=2)
    ax.plot(g[:, 0], g[:, 1], "--", color=col, lw=1.3, alpha=0.55, zorder=2)
    ax.scatter(p[-1, 0], p[-1, 1], s=16, color=col, alpha=0.8, zorder=3)            # present (t=C)

# Ball: faint orange K modes underneath, then bold context + GT on top.
for k in range(kpreds_ft.shape[2]):
    m = to_display(kpreds_ft[idx, BALL, k])
    ax.plot(m[:, 0], m[:, 1], "-", color="#FB8C00", lw=1.0, alpha=0.45, zorder=4)
pb = to_display(past_ft[idx, BALL]); gb = to_display(tgt_ft[idx, BALL])
ax.plot(pb[:, 0], pb[:, 1], "-", color="k", lw=2.4, zorder=6)                      # ball context
ax.plot([pb[-1, 0], gb[0, 0]], [pb[-1, 1], gb[0, 1]], "--", color="k", lw=2.4, zorder=6)
ax.plot(gb[:, 0], gb[:, 1], "--", color="k", lw=2.4, zorder=6)                     # ball GT future
ax.scatter(pb[-1, 0], pb[-1, 1], s=60, color="k", zorder=7,                        # ball present (t=C)
           marker="o", edgecolors="white", linewidths=1.2)

handles = [
    plt.Line2D([0], [0], marker="o", color="k", lw=0, markersize=7,
               markeredgecolor="white", label="position at $t{=}C$"),
    plt.Line2D([0], [0], color="k", lw=2.2, ls="-", label="ball: past"),
    plt.Line2D([0], [0], color="k", lw=2.2, ls="--", label="ball: actual future"),
    plt.Line2D([0], [0], color="#FB8C00", lw=1.4, alpha=0.7, label="ball: 20 predicted modes"),
    plt.Line2D([0], [0], color="#E53935", lw=1.4, alpha=0.7, label="Team A"),
    plt.Line2D([0], [0], color="#1E88E5", lw=1.4, alpha=0.7, label="Team B"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False, fontsize=8)
ax.set_title("Players move predictably; the ball fans across plausible futures")
fig.tight_layout()
fig.savefig(OUT / "fig4.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[fig4] saved {OUT/'fig4.png'}")
