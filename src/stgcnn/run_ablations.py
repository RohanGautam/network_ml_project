import sys
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgcnn.evaluate import evaluate

def main():
    checkpoints = {
        "1. Baseline Model": {
            "path": "src/stgcnn/best_tuned.pt",
            "tta": False,
            "clamp": False
        },
        "2. Baseline + Kinematics": {
            "path": "src/stgcnn/best_kinematics.pt",
            "tta": False,
            "clamp": False
        },
        "3. Baseline + Kinematics + Learnable Edges": {
            "path": "src/stgcnn/best_mse_tuned.pt",
            "tta": False,
            "clamp": False
        },
        "4. Baseline + Kinematics + Learnable Edges + TTA & Clamping": {
            "path": "src/stgcnn/best_mse_tuned.pt",
            "tta": True,
            "clamp": True
        }
    }
    
    results = []
    print("\n" + "="*70)
    print("RUNNING STGCNN ABLATION STUDY")
    print("="*70 + "\n")
    
    for name, config in checkpoints.items():
        path = str(PROJECT_ROOT / config["path"])
        if not os.path.exists(path):
            print(f"Checkpoint not found for {name}: {config['path']}")
            print("Please ensure you have trained/copied this checkpoint to the cluster.\n")
            results.append((name, "N/A", "N/A", "N/A", "N/A", "N/A"))
            continue
            
        print(f"Evaluating: {name} (TTA={config['tta']}, Clamp={config['clamp']})...")
        
        # Load metadata
        try:
            ckpt_data = torch.load(path, map_location="cpu")
            saved_epoch = ckpt_data.get("epoch", "Unknown")
            saved_mse = ckpt_data.get("val_mse", "Unknown")
            if isinstance(saved_mse, float):
                saved_mse_str = f"{saved_mse:.4f}"
            else:
                saved_mse_str = str(saved_mse)
            print(f"   Checkpoint Metadata -> Saved at Epoch: {saved_epoch}, Saved Val MSE: {saved_mse_str} ft²")
        except Exception as e:
            saved_epoch = "Error"
            saved_mse_str = "Error"
            print(f"Could not read checkpoint metadata: {e}")
            
        # Run evaluation
        try:
            mse, ade, fde = evaluate(path, tta=config["tta"], clamp=config["clamp"])
            results.append((name, f"{mse:.4f}", f"{ade:.4f}", f"{fde:.4f}", str(saved_epoch), saved_mse_str))
            print(f"   Success! Evaluated MSE={mse:.4f} ft²\n")
        except Exception as e:
            print(f"Error evaluating {name}: {e}\n")
            results.append((name, "ERROR", "ERROR", "ERROR", str(saved_epoch), saved_mse_str))
            
    print("\n" + "="*80)
    print("FINAL ABLATION RESULTS TABLE")
    print("="*80)
    print("| Configuration | Val MSE (ft²) | Val ADE (ft) | Val FDE (ft) | Epoch | Train-time Best MSE |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|")
    for name, mse, ade, fde, epoch, t_mse in results:
        print(f"| {name} | {mse} | {ade} | {fde} | {epoch} | {t_mse} |")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
