"""
Visualise the interaction graph inferred by EqMotion (run: eqmotion_agentid).

WandB run : vc1ecnsi  |  name: eqmotion_agentid
Checkpoint: NML_base/vc1ecnsi/checkpoints/epoch=39-step=2800.ckpt
val/ade_ft : 1.979 ft  |  val/fde_ft : 4.195 ft

Produces per-play figures (category heatmaps + court graph) and a
summary figure showing average category weights broken down by pair type
(same-team, cross-team, ball-player).

Usage:
    python src/equivariance/viz_interaction_graph.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equivariance.eqmotion_nba import NBAEqMotionLightningModel  # noqa: E402

CKPT_PATH = (
    PROJECT_ROOT / "NML_base" / "vc1ecnsi" / "checkpoints" / "epoch=39-step=2800.ckpt"
)
SPLITS_FILE = PROJECT_ROOT / "splits" / "fold0.json"
COURT_IMAGE = PROJECT_ROOT / "src" / "img" / "basketball_court.png"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CONTEXT_SIZE = 8
HORIZON_SIZE = 12
N_PLAYS = 6  # per-play figures to produce
N_SUMMARY_PLAYS = 50  # plays to average for the summary figure

TEAM_COLORS = {-1: "#E53935", 0: "#43A047", 1: "#1E88E5"}
TEAM_LABELS = {-1: "Team A", 0: "Ball", 1: "Team B"}
# Edge colours for the two learned interaction categories
CAT_COLORS = ["#FF7043", "#5C6BC0"]  # cat-0 = orange, cat-1 = indigo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_display(xy: np.ndarray) -> np.ndarray:
    """Shift centred coords → court display coords ([0,95] × [0,50])."""
    out = xy.copy()
    out[..., 0] += 47.5
    out[..., 1] += 25.0
    return out


def setup_court(ax, title: str = "") -> None:
    court = plt.imread(str(COURT_IMAGE))
    ax.imshow(court, extent=[0, 95, 0, 50], aspect="auto", zorder=0, alpha=0.5)
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 50)
    ax.set_xlabel("x (ft)", fontsize=7)
    ax.set_ylabel("y (ft)", fontsize=7)
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=8)


def agent_labels(raw_feats: torch.Tensor) -> list[str]:
    """
    raw_feats : [N, 2] — (isplayer, team)
    Returns list of N human-readable labels: 'Ball', 'A0'..'A4', 'B0'..'B4'.
    """
    labels, counts = [], {-1: 0, 1: 0}
    for i in range(raw_feats.shape[0]):
        team = int(raw_feats[i, 1].item())
        if int(raw_feats[i, 0].item()) == 0:
            labels.append("Ball")
        else:
            prefix = "A" if team == -1 else "B"
            labels.append(f"{prefix}{counts[team]}")
            counts[team] += 1
    return labels


def pair_type(team_i: int, team_j: int) -> str:
    if team_i == 0 or team_j == 0:
        return "ball-player"
    return "same-team" if team_i == team_j else "cross-team"


def compute_norm_stats(manifest: dict, data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    all_pos = []
    for f in manifest["train"]:
        seq = torch.load(data_dir / f, weights_only=False)
        all_pos.append(seq[:, :, :2])
    all_pos = torch.cat(all_pos, dim=0)
    return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))


def preprocess(seq_raw: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    seq_raw : [T, N, 4]  — [x, y, isplayer, team] (unnormalised)
    Returns  : [T, N, 6]  — [x_n, y_n, dx, dy, isplayer, team]
    """
    seq = seq_raw.clone()
    seq[:, :, :2] = (seq[:, :, :2] - mu) / sigma
    vel = torch.zeros_like(seq[:, :, :2])
    vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
    return torch.cat([seq[:, :, :2], vel, seq[:, :, 2:]], dim=-1)


# ---------------------------------------------------------------------------
# Core: extract interaction categories from model
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_category(
    lm: NBAEqMotionLightningModel,
    X: torch.Tensor,   # [B, T_p, N, 6] normalised
) -> torch.Tensor:
    """
    Bypass the Lightning wrapper to get the raw category tensor [B, N, N, K].
    Replicates NBAEqMotionModel.forward input preparation then calls EqMotion directly.
    """
    eqm = lm.net.model  # EqMotion
    B, T, N, _ = X.shape
    pos = X[:, :, :, :2].permute(0, 2, 1, 3)   # [B, N, T, 2]
    vel = X[:, :, :, 2:4].permute(0, 2, 1, 3)  # [B, N, T, 2]
    h = torch.norm(vel, dim=-1)                  # [B, N, T]
    agent_id = X[:, 0, :, 4:]                   # [B, N, 2]
    _, cat_per_layer = eqm(h, pos, vel, agent_id=agent_id)
    # cat_per_layer is a list of identical [B, N, N, K] tensors (computed once, reused)
    return cat_per_layer[0]  # [B, N, N, K]


# ---------------------------------------------------------------------------
# Figure 1: per-play — category heatmaps + court graph
# ---------------------------------------------------------------------------


def fig_play(
    seq_raw: torch.Tensor,    # [T, N, 4] unnormalised
    category: torch.Tensor,   # [N, N, K]
    play_idx: int,
    fname: str,
) -> None:
    N, K = category.shape[0], category.shape[2]
    cat = category.cpu().numpy()  # [N, N, K]

    # Static agent metadata (invariant across time)
    static = seq_raw[0, :, 2:]  # [N, 2]: isplayer, team
    labels = agent_labels(static)
    teams = [int(static[i, 1].item()) for i in range(N)]
    node_colors = [TEAM_COLORS[t] for t in teams]

    # Agent positions at last context frame for the court diagram
    pos_raw = seq_raw[CONTEXT_SIZE - 1, :, :2].numpy()  # [N, 2]
    pos_disp = to_display(pos_raw)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle(
        f"Play {play_idx} — Inferred interaction graph  (eqmotion_agentid, run vc1ecnsi)",
        fontsize=10,
    )

    # ── Panels 0 & 1: N×N category heatmaps ─────────────────────────────────
    for k in range(min(K, 2)):
        ax = axes[k]
        mat = cat[:, :, k]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Target agent j", fontsize=7)
        ax.set_ylabel("Source agent i", fontsize=7)
        ax.set_title(f"c_{{ij,{k}}}  (category {k} weight)", fontsize=8)
        for i in range(N):
            for j in range(N):
                v = mat[i, j]
                fc = "black" if 0.25 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color=fc)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 2: court graph ─────────────────────────────────────────────────
    ax = axes[2]
    setup_court(ax, title="Inferred edges at context end  (opacity = certainty)")

    # Draw directed edges: colour = dominant category, opacity = certainty
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = cat[i, j]               # [K]
            dom = int(np.argmax(w))
            certainty = float(w[dom])   # in [0.5, 1.0]
            alpha = (certainty - 0.5) * 2.0  # remap to [0, 1]
            if alpha < 0.2:
                continue
            xi, yi = pos_disp[i]
            xj, yj = pos_disp[j]
            ax.annotate(
                "",
                xy=(xj, yj),
                xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=CAT_COLORS[dom],
                    alpha=alpha,
                    lw=alpha * 1.8,
                    mutation_scale=6,
                ),
                zorder=2,
            )

    # Draw nodes
    for i in range(N):
        x, y = pos_disp[i]
        is_ball = labels[i] == "Ball"
        ax.scatter(
            x, y,
            color=node_colors[i],
            s=130 if is_ball else 65,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.text(
            x, y + 1.4, labels[i],
            ha="center", va="bottom", fontsize=5.5, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc=node_colors[i], alpha=0.75, lw=0),
            zorder=6,
        )

    legend_handles = [
        mpatches.Patch(color=TEAM_COLORS[t], label=TEAM_LABELS[t]) for t in [-1, 0, 1]
    ] + [
        plt.Line2D([0], [0], color=CAT_COLORS[0], lw=2, label="Category 0"),
        plt.Line2D([0], [0], color=CAT_COLORS[1], lw=2, label="Category 1"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right")

    fig.tight_layout()
    out = FIGURES_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: summary — average categories + breakdown by pair type
# ---------------------------------------------------------------------------


def fig_summary(
    categories: list[torch.Tensor],   # list of [N, N, K]
    seq_raws: list[torch.Tensor],     # list of [T, N, 4]
) -> None:
    avg_cat = torch.stack(categories).mean(0).cpu().numpy()  # [N, N, K]
    N, K = avg_cat.shape[0], avg_cat.shape[2]
    labels = agent_labels(seq_raws[0][0, :, 2:])

    # Pair-type breakdown: collect per-category mean for each pair type
    pair_buckets: dict[str, list[float]] = {
        "ball-player": [],
        "same-team": [],
        "cross-team": [],
    }
    for cat, seq in zip(categories, seq_raws):
        cat_np = cat.cpu().numpy()
        teams = [int(seq[0, i, 3].item()) for i in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pt = pair_type(teams[i], teams[j])
                # Use category-0 weight as the representative scalar
                # (cat-1 = 1 - cat-0, so one is sufficient)
                pair_buckets[pt].append(float(cat_np[i, j, 0]))

    pair_means = {k: np.mean(v) for k, v in pair_buckets.items()}
    pair_stds = {k: np.std(v) for k, v in pair_buckets.items()}

    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(
        f"Interaction graph summary over {len(categories)} val plays  (eqmotion_agentid)",
        fontsize=10,
    )
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # ── Panel 0 & 1: average heatmaps ───────────────────────────────────────
    for k in range(min(K, 2)):
        ax = fig.add_subplot(gs[0, k])
        mat = avg_cat[:, :, k]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Target j", fontsize=7)
        ax.set_ylabel("Source i", fontsize=7)
        ax.set_title(f"Mean c_{{ij,{k}}} across {len(categories)} plays", fontsize=8)
        for i in range(N):
            for j in range(N):
                v = mat[i, j]
                fc = "black" if 0.25 < v < 0.75 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color=fc)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 2: bar chart — pair-type breakdown ─────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    pair_colors = {"ball-player": TEAM_COLORS[0], "same-team": "#9C27B0", "cross-team": "#FF7043"}
    keys = list(pair_means.keys())
    vals = [pair_means[k] for k in keys]
    errs = [pair_stds[k] for k in keys]
    bars = ax.bar(
        keys, vals, yerr=errs,
        color=[pair_colors[k] for k in keys],
        alpha=0.8, width=0.5,
        error_kw=dict(elinewidth=1, capsize=4, ecolor="black"),
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Uniform (0.5)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean category-0 weight", fontsize=8)
    ax.set_title("Category-0 weight by pair type\n(deviations from 0.5 = specialisation)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7)

    # Annotate bars
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7,
        )

    out = FIGURES_DIR / "interaction_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CKPT_PATH.relative_to(PROJECT_ROOT)}")
    lm = NBAEqMotionLightningModel.load_from_checkpoint(
        str(CKPT_PATH), map_location=device, strict=False
    )
    lm.eval()
    print("  OK\n")

    # ── Data setup ────────────────────────────────────────────────────────────
    manifest = json.loads(SPLITS_FILE.read_text())
    data_dir = PROJECT_ROOT / manifest["data_dir"]

    print("Computing normalisation stats from train set...")
    mu, sigma = compute_norm_stats(manifest, data_dir)
    print(f"  mu={mu.tolist()}  sigma={sigma.tolist()}\n")

    # Collect val sequences long enough for one context window
    val_seqs = []
    for fname in manifest["val"]:
        seq = torch.load(data_dir / fname, weights_only=False)
        if seq.shape[0] >= CONTEXT_SIZE + HORIZON_SIZE:
            val_seqs.append(seq)

    print(f"Usable val sequences: {len(val_seqs)}\n")

    # ── Per-play figures ──────────────────────────────────────────────────────
    all_categories: list[torch.Tensor] = []
    all_raws: list[torch.Tensor] = []

    n_to_process = max(N_PLAYS, N_SUMMARY_PLAYS)
    for idx, seq_raw in enumerate(val_seqs[:n_to_process]):
        seq_proc = preprocess(seq_raw, mu, sigma)
        X = seq_proc[:CONTEXT_SIZE].unsqueeze(0).to(device)  # [1, T_p, N, 6]
        cat = extract_category(lm, X)[0]                      # [N, N, K]
        all_categories.append(cat)
        all_raws.append(seq_raw)

        if idx < N_PLAYS:
            print(f"Play {idx}: T={seq_raw.shape[0]}  N={seq_raw.shape[1]}")
            fig_play(seq_raw, cat, idx, f"interaction_play{idx:02d}.png")

    # ── Summary figure ────────────────────────────────────────────────────────
    print(f"\nGenerating summary over {len(all_categories)} plays...")
    fig_summary(all_categories, all_raws)

    print(f"\nAll figures saved to {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
