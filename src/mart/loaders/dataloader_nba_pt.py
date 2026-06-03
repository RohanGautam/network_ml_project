"""Dataset adapter that feeds MART from network_ml_project's .pt files.

Mirrors GroupNet/groupnet/data/dataloader_nba_pt.py so the two pipelines can be
compared head-to-head, but emits the (past, future) tuple shape MART's
main_nba.py expects: ([N, T_p, 2], [N, T_f, 2]).

Pipeline:
    1. Resolve a split manifest (splits/<name>.json) into train/val file lists.
    2. Compute z-score (mu, sigma) over (x, y) from training files only.
    3. Per file: load [T, 11, F], reorder agents into the canonical
       [TeamA(5), TeamB(5), Ball(1)] layout, z-score (x, y).
    4. On __getitem__: window of length past+future starting at `start`,
       returned as (past[N, T_p, 2], future[N, T_f, 2], agent_ids[N]).

`agent_ids` is the canonical embedding-index tensor [0]*5 + [1]*5 + [2]
(TeamA=0, TeamB=1, Ball=2). Stock MART can ignore it; MART_ID consumes it
through its nn.Embedding (mirrors GroupNetWithID).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


ENTITY_ID_COL = -1   # last feature column holds entity id (matches baseline)

# After canonical reorder the IDs are fixed for every NBA scene.
# Order: 5 TeamA (id=0), 5 TeamB (id=1), 1 Ball (id=2).
CANONICAL_AGENT_IDS = torch.tensor([0] * 5 + [1] * 5 + [2], dtype=torch.long)


def load_split_files(manifest_path):
    """Resolve a network_ml_project split manifest into (train, val) file lists.

    The manifest stores `data_dir` relative to the network_ml_project root.
    We assume `manifest_path` lives at <project_root>/splits/<name>.json.
    """
    manifest_path = Path(manifest_path).resolve()
    project_root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())
    data_dir = project_root / manifest["data_dir"]
    train = [data_dir / f for f in manifest["train"]]
    val = [data_dir / f for f in manifest["val"]]
    return train, val


def compute_xy_stats(files, iso=False):
    """Mean/std over (x, y), pooled across all files and timesteps.

    iso=False: per-axis std (anisotropic, default — legacy behaviour).
    iso=True:  a single shared scalar std for both axes. This is a prerequisite
        for rotation/reflection augmentation: under anisotropic per-axis scaling
        a physical rotation becomes a *shear* in the normalized frame the model
        sees, so the augmented samples are geometrically inconsistent. A shared
        scalar makes a rotation in raw feet map to a rotation in normed space
        (same trick EqMotion needed for its O(2) equivariance).
    """
    chunks = []
    for f in files:
        seq = torch.load(f).float()  # [T, N, F]
        chunks.append(seq[:, :, :2])
    pooled = torch.cat(chunks, dim=0)
    mu = pooled.mean(dim=(0, 1))
    if iso:
        s = pooled.std()
        sigma = torch.stack([s, s])
    else:
        sigma = pooled.std(dim=(0, 1))
    return mu, sigma


def _reorder_to_canonical(seq):
    """Reorder agents into [TeamA(5), TeamB(5), Ball(1)] using the entity-id col."""
    ids = seq[0, :, ENTITY_ID_COL].long()
    team_a = (ids == -1).nonzero(as_tuple=True)[0]
    team_b = (ids == 1).nonzero(as_tuple=True)[0]
    ball = (ids == 0).nonzero(as_tuple=True)[0]
    if len(team_a) != 5 or len(team_b) != 5 or len(ball) != 1:
        raise ValueError(
            f"Expected 5 TeamA / 5 TeamB / 1 Ball, "
            f"got {len(team_a)}/{len(team_b)}/{len(ball)} (ids={ids.tolist()})"
        )
    new_order = torch.cat([team_a, team_b, ball])
    return seq.index_select(1, new_order)


class MARTNBAPTDataset(Dataset):
    """Yields (past[N, T_p, 2], future[N, T_f, 2], agent_ids[N]) windows for MART."""

    def __init__(self, files, mu, sigma, past_length=8, future_length=12):
        super().__init__()
        self.past_length = past_length
        self.future_length = future_length
        self.window = past_length + future_length

        self.mu = mu.view(1, 1, 2).float()
        self.sigma = sigma.view(1, 1, 2).float()

        self.sequences = []   # each: [T, 11, 2] z-scored
        self.max_start = []

        for f in files:
            seq = torch.load(f).float()  # [T, 11, F]
            if seq.shape[0] < self.window:
                continue
            seq = _reorder_to_canonical(seq)
            xy = (seq[:, :, :2] - self.mu) / self.sigma
            self.sequences.append(xy.contiguous())
            self.max_start.append(seq.shape[0] - self.window)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_idx, start = index
        T = self.window
        xy = self.sequences[seq_idx][start:start + T]   # [T, 11, 2]
        xy = xy.permute(1, 0, 2).contiguous()            # [11, T, 2]
        past = xy[:, :self.past_length]                  # [11, T_p, 2]
        future = xy[:, self.past_length:]                # [11, T_f, 2]
        return past, future, CANONICAL_AGENT_IDS


class WindowSampler(Sampler):
    """One (seq_idx, random_start) per sequence per epoch.

    Mirrors network_ml_project's NBASampler so the train budget per epoch
    matches the GroupNet adapter exactly.
    """

    def __init__(self, batch_size, max_start, seed=0, shuffle=True):
        self.batch_size = batch_size
        self.max_start = max_start
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n = len(self)
        if self.shuffle:
            perm = torch.randperm(n, generator=self.generator).tolist()
        else:
            perm = list(range(n))
        for i in perm:
            ms = self.max_start[i]
            start = torch.randint(0, ms + 1, size=(), generator=self.generator).item()
            yield i, start

    def __len__(self):
        return len(self.max_start)


class WindowEvalSampler(Sampler):
    """Deterministic, multi-window validation sampler (mirrors EqMotion's
    NBAEvalSampler exactly so val/mse_ft is directly comparable across models).

    Instead of one random window per sequence per epoch (the noisy
    `WindowSampler` behaviour, which makes val bounce between epochs and gives
    a lucky-draw checkpoint signal), this enumerates a fixed set of evenly
    spaced windows per sequence, capped at `windows_per_seq`. Each sequence
    contributes the same number of windows (no over-weighting of long ones).
    """

    def __init__(self, max_start, windows_per_seq=8):
        self.windows = []
        for i, ms in enumerate(max_start):
            if ms <= 0:
                starts = [0]
            else:
                k = min(windows_per_seq, ms + 1)
                starts = sorted(
                    {int(round(s)) for s in torch.linspace(0, ms, k).tolist()}
                )
            self.windows.extend((i, s) for s in starts)

    def __iter__(self):
        return iter(self.windows)

    def __len__(self):
        return len(self.windows)
