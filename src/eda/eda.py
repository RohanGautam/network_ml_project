"""
Exploratory data analysis for the NBA trajectory dataset.
Saves figures to src/eda/figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = PROJECT_ROOT / "splits" / "fold0.json"
COURT_IMAGE = PROJECT_ROOT / "src" / "img" / "basketball_court.png"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TEAM_COLORS = {-1: "#E53935", 0: "#43A047", 1: "#1E88E5"}
TEAM_LABELS = {-1: "Team A", 0: "Ball", 1: "Team B"}

CONTEXT_SIZE = 8
HORIZON_SIZE = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_train_sequences():
    manifest = json.loads(SPLITS_FILE.read_text())
    files = [PROJECT_ROOT / manifest["data_dir"] / f for f in manifest["train"]]
    print(f"Loading {len(files)} train sequences...")
    seqs = [torch.load(f, weights_only=False) for f in files]
    print("Done.")
    return seqs


def to_display(xy: np.ndarray) -> np.ndarray:
    """Shift centered coords → court display coords ([0,95] x [0,50])."""
    out = xy.copy()
    out[:, 0] += 47.5
    out[:, 1] += 25.0
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


# ---------------------------------------------------------------------------
# Figure 1: Sequence length distribution
# ---------------------------------------------------------------------------


def fig_sequence_lengths(seqs):
    lengths = [s.shape[0] for s in seqs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=60, color="#1E88E5", edgecolor="white", linewidth=0.4)
    ax.axvline(
        CONTEXT_SIZE + HORIZON_SIZE,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Window size C+H = {CONTEXT_SIZE + HORIZON_SIZE}",
    )
    ax.axvline(
        np.median(lengths),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {np.median(lengths):.0f}",
    )
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Length Distribution")
    ax.legend()
    stats = (
        f"n={len(lengths)}  min={min(lengths)}  "
        f"max={max(lengths)}  mean={np.mean(lengths):.1f}  "
        f"median={np.median(lengths):.0f}"
    )
    ax.text(
        0.98,
        0.95,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sequence_lengths.png", dpi=150)
    plt.close(fig)
    print("Saved 01_sequence_lengths.png")


# ---------------------------------------------------------------------------
# Figure 2: Spatial heatmaps by agent type
# ---------------------------------------------------------------------------


def fig_spatial_heatmaps(seqs):
    # Collect positions per team
    positions = {-1: [], 0: [], 1: []}
    for seq in seqs:
        for team_val in [-1, 0, 1]:
            mask = seq[0, :, 3] == team_val  # static across time
            xy = seq[:, mask, :2].reshape(-1, 2).numpy()
            positions[team_val].append(xy)
    for k in positions:
        positions[k] = np.concatenate(positions[k], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = {-1: "Team A Players", 0: "Ball", 1: "Team B Players"}
    cmaps = {-1: "Reds", 0: "Greens", 1: "Blues"}

    for ax, team_val in zip(axes, [-1, 0, 1]):
        setup_court(ax, title=titles[team_val])
        xy_disp = to_display(positions[team_val])
        h, _, _ = np.histogram2d(
            xy_disp[:, 0], xy_disp[:, 1], bins=[95, 50], range=[[0, 95], [0, 50]]
        )
        h = h / h.max()
        h_masked = np.ma.masked_where(h < 0.05, h)
        im = ax.imshow(
            h_masked.T,
            extent=[0, 95, 0, 50],
            origin="lower",
            cmap=cmaps[team_val],
            alpha=0.65,
            aspect="auto",
            vmin=0,
            vmax=1,
            zorder=1,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Relative density", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Spatial Position Heatmaps (train set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_position_heatmaps.png", dpi=150)
    plt.close(fig)
    print("Saved 02_position_heatmaps.png")


# ---------------------------------------------------------------------------
# Figure 3: Sample trajectories on court
# ---------------------------------------------------------------------------


def fig_sample_trajectories(seqs, n=6):
    # Pick sequences spread across the length distribution
    lengths = np.array([s.shape[0] for s in seqs])
    percentiles = np.linspace(10, 90, n)
    indices = [
        np.argmin(np.abs(lengths - np.percentile(lengths, p))) for p in percentiles
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        seq = seqs[idx].numpy()  # [T, N, 4]
        T = seq.shape[0]
        setup_court(ax, title=f"Sequence {idx}  (T={T})")

        for n_idx in range(seq.shape[1]):
            team_val = int(seq[0, n_idx, 3])
            color = TEAM_COLORS[team_val]
            xy = to_display(seq[:, n_idx, :2])
            # Context (solid) vs horizon (dashed) if long enough
            if T >= CONTEXT_SIZE + HORIZON_SIZE:
                ax.plot(
                    xy[:CONTEXT_SIZE, 0],
                    xy[:CONTEXT_SIZE, 1],
                    color=color,
                    linewidth=1.5,
                    zorder=2,
                )
                ax.plot(
                    xy[CONTEXT_SIZE : CONTEXT_SIZE + HORIZON_SIZE, 0],
                    xy[CONTEXT_SIZE : CONTEXT_SIZE + HORIZON_SIZE, 1],
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                    zorder=2,
                )
                ax.scatter(*xy[CONTEXT_SIZE - 1], color=color, s=30, zorder=3)
            else:
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.2, zorder=2)
            ax.scatter(*xy[0], marker="o", color=color, s=20, zorder=3)

        # Legend for one panel only
        if ax is axes[0]:
            patches = [
                mpatches.Patch(color=TEAM_COLORS[t], label=TEAM_LABELS[t])
                for t in [-1, 0, 1]
            ]
            patches += [
                plt.Line2D([0], [0], color="gray", linewidth=1.5, label="Context"),
                plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=1.5,
                    linestyle="--",
                    label="Horizon",
                ),
            ]
            ax.legend(handles=patches, fontsize=7, loc="upper right")

    fig.suptitle("Sample Trajectories (solid = context, dashed = horizon)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sample_trajectories.png", dpi=150)
    plt.close(fig)
    print("Saved 03_sample_trajectories.png")


# ---------------------------------------------------------------------------
# Figure 4: Per-step displacement distribution by agent type
# ---------------------------------------------------------------------------


def fig_speed_distribution(seqs):
    disps = {-1: [], 0: [], 1: []}
    for seq in seqs:
        if seq.shape[0] < 2:
            continue
        step_disp = torch.norm(seq[1:, :, :2] - seq[:-1, :, :2], dim=-1)  # [T-1, N]
        for team_val in [-1, 0, 1]:
            mask = seq[0, :, 3] == team_val
            disps[team_val].append(step_disp[:, mask].flatten())
    for k in disps:
        disps[k] = torch.cat(disps[k]).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    for team_val in [-1, 0, 1]:
        ax.hist(
            disps[team_val],
            bins=80,
            range=(0, 5),
            color=TEAM_COLORS[team_val],
            alpha=0.6,
            label=TEAM_LABELS[team_val],
            density=True,
        )
    ax.set_xlabel("Per-step displacement (ft/frame)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Step Displacement Distribution")
    ax.legend()

    # Box plot
    ax = axes[1]
    data = [disps[t] for t in [-1, 0, 1]]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, team_val in zip(bp["boxes"], [-1, 0, 1]):
        patch.set_facecolor(TEAM_COLORS[team_val])
        patch.set_alpha(0.7)
    ax.set_xticklabels([TEAM_LABELS[t] for t in [-1, 0, 1]])
    ax.set_ylabel("Per-step displacement (ft/frame)")
    ax.set_title("Per-Step Displacement (no outliers)")

    fig.suptitle("Agent Speed Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_speed_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved 04_speed_distribution.png")


# ---------------------------------------------------------------------------
# Figure 5: Horizon displacement — task difficulty
# ---------------------------------------------------------------------------


def fig_horizon_displacement(seqs):
    """How far does each agent move over the H=12 prediction horizon?"""
    horizon_disps = {-1: [], 0: [], 1: []}
    for seq in seqs:
        T = seq.shape[0]
        if T < CONTEXT_SIZE + HORIZON_SIZE:
            continue
        context_end = CONTEXT_SIZE
        horizon_end = CONTEXT_SIZE + HORIZON_SIZE
        for team_val in [-1, 0, 1]:
            mask = seq[0, :, 3] == team_val
            start_pos = seq[context_end, mask, :2]  # position at context end
            end_pos = seq[horizon_end - 1, mask, :2]  # position at horizon end
            disp = torch.norm(end_pos - start_pos, dim=-1)  # [n_agents]
            horizon_disps[team_val].append(disp)
    for k in horizon_disps:
        horizon_disps[k] = torch.cat(horizon_disps[k]).numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    for team_val in [-1, 0, 1]:
        d = horizon_disps[team_val]
        ax.hist(
            d,
            bins=60,
            range=(0, 30),
            color=TEAM_COLORS[team_val],
            alpha=0.6,
            label=f"{TEAM_LABELS[team_val]} (μ={d.mean():.1f} ft)",
            density=True,
        )
    ax.set_xlabel(f"Total displacement over H={HORIZON_SIZE} frames (ft)")
    ax.set_ylabel("Density")
    ax.set_title("Horizon Displacement Distribution (task difficulty)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_horizon_displacement.png", dpi=150)
    plt.close(fig)
    print("Saved 05_horizon_displacement.png")


# ---------------------------------------------------------------------------
# Figure 6: Inter-agent distance distribution (motivation for GNN)
# ---------------------------------------------------------------------------


def fig_inter_agent_distances(seqs):
    """
    Pairwise distances between agents — motivates graph-based interaction modeling.
    Shows that agents are consistently close enough to influence each other.
    """
    same_team_dists = []
    cross_team_dists = []
    player_ball_dists = []

    for seq in seqs[:500]:  # sample for speed
        T, N, _ = seq.shape
        for t in range(0, T, 5):  # every 5 frames
            frame = seq[t]  # [N, 4]
            for i in range(N):
                for j in range(i + 1, N):
                    d = torch.norm(frame[i, :2] - frame[j, :2]).item()
                    ti, tj = frame[i, 3].item(), frame[j, 3].item()
                    if ti == 0 or tj == 0:
                        player_ball_dists.append(d)
                    elif ti == tj:
                        same_team_dists.append(d)
                    else:
                        cross_team_dists.append(d)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 80, 60)
    ax.hist(
        same_team_dists,
        bins=bins,
        alpha=0.6,
        density=True,
        color="#9C27B0",
        label=f"Same team (μ={np.mean(same_team_dists):.1f} ft)",
    )
    ax.hist(
        cross_team_dists,
        bins=bins,
        alpha=0.6,
        density=True,
        color="#FF7043",
        label=f"Cross team (μ={np.mean(cross_team_dists):.1f} ft)",
    )
    ax.hist(
        player_ball_dists,
        bins=bins,
        alpha=0.6,
        density=True,
        color="#43A047",
        label=f"Player–ball (μ={np.mean(player_ball_dists):.1f} ft)",
    )
    ax.set_xlabel("Pairwise distance (ft)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Inter-Agent Distance Distribution\n(motivation for interaction modeling)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_inter_agent_distances.png", dpi=150)
    plt.close(fig)
    print("Saved 06_inter_agent_distances.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seqs = load_train_sequences()

    fig_sequence_lengths(seqs)
    fig_spatial_heatmaps(seqs)
    fig_sample_trajectories(seqs)
    fig_speed_distribution(seqs)
    fig_horizon_displacement(seqs)
    fig_inter_agent_distances(seqs)

    print(f"\nAll figures saved to {FIGURES_DIR}")
