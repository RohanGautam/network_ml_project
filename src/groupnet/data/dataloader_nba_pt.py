"""Dataset adapter that feeds GroupNet from network_ml_project's .pt files.

Pipeline (matches network_ml_project conventions):
    1. Resolve a split manifest (splits/fold0.json) into train/val file lists.
    2. Compute z-score (mu, sigma) over (x, y) from training files only.
    3. Per file: load [T, 11, F] tensor, reorder agents into the canonical
       [TeamA(5), TeamB(5), Ball(1)] layout that GroupNet's add_category
       expects, normalize (x, y) with the training stats, store agent ids.
    4. On __getitem__: random window of length past+future, return
       (past[N,T_p,2], future[N,T_f,2], agent_ids[N]).

Coordinates stay court-centered (no extra translation); only z-score is applied.
Entity-id mapping (network_ml_project -> embedding index):
    -1 (TeamA) -> 0,  +1 (TeamB) -> 1,  0 (Ball) -> 2.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


RAW_TO_EMBED = {-1: 0, 1: 1, 0: 2}
ENTITY_ID_COL = -1   # last feature column holds entity id (matches baseline)


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


def compute_xy_stats(files):
    """Per-axis mean/std over (x, y), pooled across all files and timesteps."""
    chunks = []
    for f in files:
        seq = torch.load(f).float()  # [T, N, F]
        chunks.append(seq[:, :, :2])
    pooled = torch.cat(chunks, dim=0)  # [sum_T, N, 2]
    mu = pooled.mean(dim=(0, 1))       # [2]
    sigma = pooled.std(dim=(0, 1))     # [2]
    return mu, sigma


def _reorder_to_canonical(seq):
    """Reorder agents into [TeamA(5), TeamB(5), Ball(1)] using the entity-id col.

    Args:
        seq: [T, 11, F] float tensor.
    Returns:
        reordered: [T, 11, F] float tensor.
        embed_idx: [11] long tensor of embedding indices (matches the new order).
    """
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
    reordered = seq.index_select(1, new_order)
    embed_idx = torch.tensor([0] * 5 + [1] * 5 + [2], dtype=torch.long)
    return reordered, embed_idx


class GroupNetNBAPTDataset(Dataset):
    """Yields (past, future, agent_ids) windows for GroupNet training."""

    def __init__(self, files, mu, sigma, past_length=8, future_length=12):
        super().__init__()
        self.past_length = past_length
        self.future_length = future_length
        self.window = past_length + future_length

        self.mu = mu.view(1, 1, 2).float()
        self.sigma = sigma.view(1, 1, 2).float()

        self.sequences = []   # each: [T, 11, 2] z-scored
        self.agent_ids = []   # each: [11] long
        self.max_start = []

        for f in files:
            seq = torch.load(f).float()  # [T, 11, F]
            if seq.shape[0] < self.window:
                # Skip files too short to yield even one full window.
                continue
            seq, embed_idx = _reorder_to_canonical(seq)
            xy = (seq[:, :, :2] - self.mu) / self.sigma
            self.sequences.append(xy.contiguous())
            self.agent_ids.append(embed_idx)
            self.max_start.append(seq.shape[0] - self.window)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_idx, start = index
        T = self.window
        xy = self.sequences[seq_idx][start:start + T]   # [T, 11, 2]
        ids = self.agent_ids[seq_idx]                    # [11]
        xy = xy.permute(1, 0, 2).contiguous()            # [11, T, 2]
        past = xy[:, :self.past_length]                  # [11, T_p, 2]
        future = xy[:, self.past_length:]                # [11, T_f, 2]
        return past, future, ids


def groupnet_collate(batch):
    """Collate to GroupNet's expected dict (extended with agent_ids)."""
    past, future, ids = zip(*batch)
    return {
        'past_traj':   torch.stack(past, dim=0),    # [B, 11, T_p, 2]
        'future_traj': torch.stack(future, dim=0),  # [B, 11, T_f, 2]
        'agent_ids':   torch.stack(ids, dim=0),     # [B, 11] long
        'seq': 'nba',
    }


class WindowSampler(Sampler):
    """One (seq_idx, random_start) per sequence per epoch.

    Mirrors network_ml_project's NBASampler: every file contributes exactly one
    randomly-positioned window per epoch. Call set_epoch(e) so each epoch uses a
    different RNG state (different starts and shuffle).
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
