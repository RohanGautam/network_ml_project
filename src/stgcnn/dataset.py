import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

class NBADataset(Dataset):
    def __init__(self, split_file: str, data_dir: str, split_key: str = "train", seq_len: int = 20, obs_len: int = 8, normalize: bool = True, use_kinematics: bool = False, data_fraction: float = 1.0):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.obs_len = obs_len
        self.normalize = normalize
        self.use_kinematics = use_kinematics
        self.data_fraction = data_fraction
        
        # Exact fold0 training split statistics
        self.register_mu = torch.tensor([0.43076536, 0.04198149])
        self.register_sigma = torch.tensor([29.67073631, 11.54754257])
        
        with open(split_file, 'r') as f:
            manifest = json.load(f)
        
        self.file_names = manifest[split_key]
        if split_key == "train" and data_fraction < 1.0:
            import random
            rng = random.Random(42)
            k = max(1, int(len(self.file_names) * data_fraction))
            self.file_names = rng.sample(self.file_names, k)
            
        self.samples = []
        self._load_data()

    def _load_data(self):
        mu = self.register_mu.float()
        sigma = self.register_sigma.float()

        for file_name in self.file_names:
            file_path = self.data_dir / file_name
            tensor = torch.load(file_path, weights_only=True).float()

            num_frames = tensor.shape[0]
            for start_idx in range(0, num_frames - self.seq_len + 1):
                window = tensor[start_idx : start_idx + self.seq_len]
                coords = window[..., :2] # [seq_len, 11, 2]

                # adjacency from raw (un-normalized) coordinates: A_ij = exp(-‖xi-xj‖)
                X_raw = coords[:self.obs_len] # [obs_len, 11, 2]
                dist = torch.cdist(X_raw, X_raw) # [obs_len, 11, 11]
                A = torch.exp(-dist)

                if self.normalize:
                    coords_norm = (coords - mu) / sigma
                else:
                    coords_norm = coords

                if self.use_kinematics:
                    vel = torch.zeros_like(coords_norm)
                    vel[1:] = coords_norm[1:] - coords_norm[:-1]
                    acc = torch.zeros_like(vel)
                    acc[1:] = vel[1:] - vel[:-1]
                    feat = torch.cat([coords_norm[:self.obs_len], vel[:self.obs_len], acc[:self.obs_len]], dim=-1) # [obs_len, 11, 6]
                    X = feat.permute(2, 0, 1) # [6, obs_len, 11]
                else:
                    X = coords_norm[:self.obs_len].permute(2, 0, 1) # [2, obs_len, 11]

                Y = coords_norm[self.obs_len:] # [pred_len, 11, 2]

                self.samples.append((X, Y, A))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]