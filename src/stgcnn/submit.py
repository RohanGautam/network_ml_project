import sys
import os
import json
import torch
import csv
from pathlib import Path
from datetime import datetime

# Add the 'src' directory to the python path
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
    
    # Load model
    model = Social_STGCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Normalization statistics
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
            
            # Compute A based on raw coordinates (vectorized)
            dist = torch.cdist(coords, coords) # [8, 11, 11]
            A = torch.exp(-dist)
                
            # Normalize coordinates
            coords_norm = (coords - mu) / sigma
            X = coords_norm.permute(2, 0, 1) # [2, 8, 11]
            
            # Add batch dimension and send to device
            X = X.unsqueeze(0).to(device) # [1, 2, 8, 11]
            A = A.unsqueeze(0).to(device) # [1, 8, 11, 11]
            
            # Predict
            mu_x, mu_y, sig_x, sig_y, rho = model(X, A)
            pred = torch.stack([mu_x, mu_y], dim=-1).squeeze(0) # [12, 11, 2]
            
            # Denormalize
            pred_denorm = pred.cpu() * sigma + mu # [12, 11, 2]
            
            # Flatten row-major
            flat_pred = pred_denorm.numpy().flatten() # [264]
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
