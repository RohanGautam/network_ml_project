import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add 'src' directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stgcnn.dataset import NBADataset
from stgcnn.model import Social_STGCNN
from utils.metrics import compute_ade, compute_fde, compute_mse

def evaluate(model_path: str, tta: bool = False, clamp: bool = False):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint first to auto-detect model parameters
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # Auto-detect parameters
    # st_gcnn1.tcn.weight shape is [hidden_dim, in_channels, kernel_size, 1]
    in_channels = state_dict["st_gcnn1.tcn.weight"].shape[1]
    hidden_dim = state_dict["st_gcnn1.tcn.weight"].shape[0]
    use_edge_importance = "st_gcnn1.edge_importance" in state_dict
    
    print(f"Auto-detected model architecture:")
    print(f"  in_channels: {in_channels}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  use_edge_importance: {use_edge_importance}")
    
    # Load dataset with matching kinematics setting
    val_dataset = NBADataset(
        split_file="splits/fold0.json", 
        data_dir="data/train/train", 
        split_key="val",
        use_kinematics=(in_channels == 6)
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Load model
    model = Social_STGCNN(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        use_edge_importance=use_edge_importance
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    mu = torch.tensor([0.43076536, 0.04198149]).to(device)
    sigma = torch.tensor([29.67073631, 11.54754257]).to(device)
    
    all_preds = []
    all_targets = []
    
    flips = [(), (0,), (1,), (0, 1)] if tta else [()]
    
    print(f"Evaluating {len(val_loader.dataset)} validation sequences...")
    print(f"Settings: TTA={tta}, Clamp={clamp}")
    
    with torch.no_grad():
        for X, Y, A in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            A = A.to(device)
            
            # Ground truth denormalized: [B, 12, 11, 2]
            Y_denorm = Y * sigma + mu
            
            preds = []
            for axes in flips:
                # Reflect input features
                X_flip = X.clone()
                for ax in axes:
                    # Negate raw coordinates for coordinate channels ax (0 or 1) and re-normalize:
                    # coords_norm_reflected = -coords_norm - 2 * mu / sigma
                    X_flip[:, ax] = -X_flip[:, ax] - 2 * mu[ax] / sigma[ax]
                    
                    if in_channels == 6:
                        # Negate velocity in that axis (ax + 2)
                        X_flip[:, ax + 2] = -X_flip[:, ax + 2]
                        # Negate acceleration in that axis (ax + 4)
                        X_flip[:, ax + 4] = -X_flip[:, ax + 4]
                
                # Predict
                mu_x, mu_y, _, _, _ = model(X_flip, A)
                pred = torch.stack([mu_x, mu_y], dim=-1) # [B, 12, 11, 2]
                
                # Denormalize
                pred_denorm = pred * sigma + mu
                
                # Un-reflect output coordinates
                for ax in axes:
                    pred_denorm[..., ax] = -pred_denorm[..., ax]
                preds.append(pred_denorm)
                
            pred_final = torch.stack(preds).mean(dim=0)
            
            if clamp:
                pred_final[..., 0].clamp_(min=-47.5, max=47.5)
                pred_final[..., 1].clamp_(min=-25.0, max=25.0)
                
            all_preds.append(pred_final.cpu())
            all_targets.append(Y_denorm.cpu())
            
    # Concatenate all batches
    all_preds_t = torch.cat(all_preds, dim=0) # [Total, 12, 11, 2]
    all_targets_t = torch.cat(all_targets, dim=0) # [Total, 12, 11, 2]
    
    # Reshape to [12, Total * 11, 2] to match metric expectation
    preds_flat = all_preds_t.permute(1, 0, 2, 3).reshape(12, -1, 2)
    targets_flat = all_targets_t.permute(1, 0, 2, 3).reshape(12, -1, 2)
    
    mse = compute_mse(preds_flat, targets_flat).item()
    ade = compute_ade(preds_flat, targets_flat).item()
    fde = compute_fde(preds_flat, targets_flat).item()
    
    print(f"\nResults:")
    print(f"  Val MSE: {mse:.4f} ft²")
    print(f"  Val ADE: {ade:.4f} ft")
    print(f"  Val FDE: {fde:.4f} ft")
    
    return mse, ade, fde

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    default_model = str(Path(__file__).resolve().parent / "best_model.pt")
    parser.add_argument("--model_path", type=str, default=default_model)
    parser.add_argument("--tta", action="store_true", help="Enable 4-way Test-Time Augmentation")
    parser.add_argument("--clamp", action="store_true", help="Enable physical court boundaries clamping")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.tta, args.clamp)
