"""MART dataloader variant that appends 2 static basket-hoop landmark nodes.

Mirrors dataloader_nba_pt.py exactly, with one addition: after reordering agents
into [TeamA(5), TeamB(5), Ball(1)], two stationary nodes are appended for the
basket hoops, giving N=13. The hoops are normalized with the same train-split
(mu, sigma) as the players so they live in the same z-scored frame the model
consumes.

Hoop positions in the court-centered frame (origin at center, x is along the
court length, y across the width):
    [-41.75, 0.0]   left basket  (5.25 ft inside the baseline at x=-47)
    [+41.75, 0.0]   right basket

Why add static landmarks: MART learns inter-agent attention purely from agent
trajectories, with no notion of where the half-court / basket / baseline are.
Two fixed nodes at the rim give every agent a positional anchor through the
relation matrix, which can help the model express basket-relative motion
(drives, passes back to the perimeter, shot release at the hoop) instead of
having to infer the court geometry from data.

API matches dataloader_nba_pt.py so callers swap the import only:
    MARTNBAPTDataset, WindowSampler, compute_xy_stats, load_split_files

Item shape changes from (past[11, T_p, 2], future[11, T_f, 2], ids[11]) to
(past[13, T_p, 2], future[13, T_f, 2], ids[13]).

Downstream caveats (NOT handled here — handle in main_nba_pt.py / eval.py):
    - Loss & val metrics in main_nba_pt.py reduce over all N agents. Including
      the 2 stationary hoops deflates train_loss / val minADE / minFDE because
      hoop "predictions" are trivially perfect, and it also dilutes the gradient
      signal away from the real 11 entities. Slice predictions/targets to the
      first 11 entities (N_REAL_AGENTS = N - N_LANDMARKS) before computing the
      loss and metrics, mirroring NBAEqMotionLightningModel.validation_step's
      n_landmarks masking.
    - MART_ID uses nn.Embedding over agent type ids. With hoops at id=3, the
      embedding's num_embeddings must be >=4. Stock MART does not consume ids
      and is unaffected.
    - submit_nba_pt.py reorders predicted agents back to the test file's layout
      using inv_order; the hoops are not in the test file, so emit predictions
      for entities [:11] only when writing the CSV.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


ENTITY_ID_COL = -1   # last feature column holds entity id (matches baseline)

# Court-centered hoop positions, in feet, before normalization. Length axis is x.
RAW_HOOPS = torch.tensor([[-41.75, 0.0], [41.75, 0.0]], dtype=torch.float32)

N_REAL_AGENTS = 11
N_LANDMARKS = RAW_HOOPS.shape[0]   # 2 baskets
HOOP_ID = 3                        # one past Ball=2

# Canonical layout after reorder + hoop append:
#   5 TeamA (id=0) | 5 TeamB (id=1) | 1 Ball (id=2) | 2 Hoops (id=3)
CANONICAL_AGENT_IDS = torch.tensor(
    [0] * 5 + [1] * 5 + [2] + [HOOP_ID] * N_LANDMARKS, dtype=torch.long,
)


def load_split_files(manifest_path):
    """Resolve a network_ml_project split manifest into (train, val) file lists.

    Identical to dataloader_nba_pt.load_split_files — replicated so this module
    is a self-contained drop-in.
    """
    manifest_path = Path(manifest_path).resolve()
    project_root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())
    data_dir = project_root / manifest["data_dir"]
    train = [data_dir / f for f in manifest["train"]]
    val = [data_dir / f for f in manifest["val"]]
    return train, val


def compute_xy_stats(files):
    """Per-axis mean/std over (x, y), pooled across files and timesteps.

    Stats are intentionally computed from the *real* trajectories only (no
    hoop augmentation) — adding the two static landmarks would bias mu toward
    0 and shrink sigma_x, distorting normalization for the real agents.
    """
    chunks = []
    for f in files:
        seq = torch.load(f).float()  # [T, N, F]
        chunks.append(seq[:, :, :2])
    pooled = torch.cat(chunks, dim=0)
    mu = pooled.mean(dim=(0, 1))
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
    """Yields (past[N, T_p, 2], future[N, T_f, 2], agent_ids[N]) windows, where
    N = 11 real entities + 2 basket landmarks = 13."""

    def __init__(self, files, mu, sigma, past_length=8, future_length=12):
        super().__init__()
        self.past_length = past_length
        self.future_length = future_length
        self.window = past_length + future_length

        self.mu = mu.view(1, 1, 2).float()
        self.sigma = sigma.view(1, 1, 2).float()

        # Normalize hoop coordinates once, broadcast over time when slicing.
        norm_hoops = (RAW_HOOPS - mu.view(1, 2)) / sigma.view(1, 2)   # [2, 2]
        self.norm_hoops = norm_hoops.float()                          # [2, 2]

        self.sequences = []   # each: [T, 13, 2] z-scored
        self.max_start = []

        for f in files:
            seq = torch.load(f).float()  # [T, 11, F]
            if seq.shape[0] < self.window:
                continue
            seq = _reorder_to_canonical(seq)
            xy = (seq[:, :, :2] - self.mu) / self.sigma     # [T, 11, 2]

            # Append static hoop nodes: [T, 11, 2] -> [T, 13, 2].
            T = xy.shape[0]
            hoop_xy = self.norm_hoops.unsqueeze(0).expand(T, -1, -1)  # [T, 2, 2]
            xy = torch.cat([xy, hoop_xy], dim=1)            # [T, 13, 2]

            self.sequences.append(xy.contiguous())
            self.max_start.append(xy.shape[0] - self.window)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq_idx, start = index
        T = self.window
        xy = self.sequences[seq_idx][start:start + T]   # [T, 13, 2]
        xy = xy.permute(1, 0, 2).contiguous()            # [13, T, 2]
        past = xy[:, :self.past_length]                  # [13, T_p, 2]
        future = xy[:, self.past_length:]                # [13, T_f, 2]
        return past, future, CANONICAL_AGENT_IDS


class WindowSampler(Sampler):
    """One (seq_idx, random_start) per sequence per epoch. Same contract as
    dataloader_nba_pt.WindowSampler — replicated so this module is a drop-in."""

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
