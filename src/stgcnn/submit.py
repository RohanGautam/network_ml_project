import sys
import os
import json
import torch
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stgcnn.model import Social_STGCNN

def generate_submission(model_path: str, test_dir: str, output_csv: str):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # auto-detect architecture from the weight shapes
    in_channels = state_dict["st_gcnn1.tcn.weight"].shape[1]
    hidden_dim = state_dict["st_gcnn1.tcn.weight"].shape[0]
    use_edge_importance = "st_gcnn1.edge_importance" in state_dict
    
    print(f"Auto-detected model architecture:")
    print(f"  in_channels: {in_channels}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  use_edge_importance: {use_edge_importance}")
    
    model = Social_STGCNN(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        use_edge_importance=use_edge_importance
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mu = torch.tensor([0.43076536, 0.04198149])
    sigma = torch.tensor([29.67073631, 11.54754257])
    
    test_dir = Path(test_dir)
    files = sorted([f for f in test_dir.iterdir() if f.name.endswith(".pt")], key=lambda x: int(x.stem))
    
    all_traj = []
    print(f"Processing {len(files)} test sequences...")
    
    with torch.no_grad():
        for f_path in files:
            seq_id = int(f_path.stem)
            tensor = torch.load(f_path, weights_only=True).float() # [8, 11, 4]
            coords = tensor[..., :2] # [8, 11, 2]

            # adjacency from raw (un-normalized) coordinates
            dist = torch.cdist(coords, coords) # [8, 11, 11]
            A = torch.exp(-dist)

            coords_norm = (coords - mu) / sigma

            if in_channels == 6:
                vel = torch.zeros_like(coords_norm)
                vel[1:] = coords_norm[1:] - coords_norm[:-1]
                acc = torch.zeros_like(vel)
                acc[1:] = vel[1:] - vel[:-1]
                feat = torch.cat([coords_norm, vel, acc], dim=-1) # [8, 11, 6]
                X = feat.permute(2, 0, 1) # [6, 8, 11]
            else:
                X = coords_norm.permute(2, 0, 1) # [2, 8, 11]

            X = X.unsqueeze(0).to(device) # [1, C, 8, 11]
            A = A.unsqueeze(0).to(device) # [1, 8, 11, 11]

            mu_x, mu_y, _, _, _ = model(X, A)
            pred = torch.stack([mu_x, mu_y], dim=-1).squeeze(0) # [12, 11, 2]
            pred_denorm = pred.cpu() * sigma + mu # [12, 11, 2]

            flat_pred = pred_denorm.numpy().flatten() # row-major [264]
            all_traj.append([seq_id] + flat_pred.tolist())
            
    columns = ["id"] + [
        f"entity_{i}_time_{t}_{axis}"
        for t in range(12)
        for i in range(11)
        for axis in ["x", "y"]
    ]
    
    all_traj.sort(key=lambda x: x[0])
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(all_traj)
    print(f"Submission saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    default_model = str(Path(__file__).resolve().parent / "best_model.pt")
    parser.add_argument("--model_path", type=str, default=default_model)
    parser.add_argument("--test_dir", type=str, default="data/test/test")
    parser.add_argument("--output_csv", type=str, default="submissions/submission_stgcnn.csv")
    args = parser.parse_args()
    
    generate_submission(args.model_path, args.test_dir, args.output_csv)
