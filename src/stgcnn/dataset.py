import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

class NBADataset(Dataset):
    def __init__(self, split_file: str, data_dir: str, split_key: str = "train", seq_len: int = 20, obs_len: int = 8, normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.obs_len = obs_len
        self.normalize = normalize
        
        # Exact fold0 training split statistics
        self.register_mu = torch.tensor([0.43076536, 0.04198149])
        self.register_sigma = torch.tensor([29.67073631, 11.54754257])
        
        with open(split_file, 'r') as f:
            manifest = json.load(f)
        
        self.file_names = manifest[split_key]
        self.samples = []
        self._load_data()

    def _load_data(self):
        # Pre-convert registers to float32
        mu = self.register_mu.float()
        sigma = self.register_sigma.float()
        
        for file_name in self.file_names:
            file_path = self.data_dir / file_name
            tensor = torch.load(file_path, weights_only=True).float()
            
            num_frames = tensor.shape[0]
            for start_idx in range(0, num_frames - self.seq_len + 1):
                window = tensor[start_idx : start_idx + self.seq_len]
                coords = window[..., :2] # [seq_len, 11, 2]
                
                # Compute A based on raw coordinates (vectorized cdist)
                X_raw = coords[:self.obs_len] # [obs_len, 11, 2]
                dist = torch.cdist(X_raw, X_raw) # [obs_len, 11, 11]
                A = torch.exp(-dist)
                
                # Normalize coordinates
                if self.normalize:
                    coords_norm = (coords - mu) / sigma
                else:
                    coords_norm = coords
                    
                X = coords_norm[:self.obs_len].permute(2, 0, 1) # [2, obs_len, 11]
                Y = coords_norm[self.obs_len:] # [pred_len, 11, 2]
                
                self.samples.append((X, Y, A))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]