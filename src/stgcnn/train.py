import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add the 'src' directory to the python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stgcnn.dataset import NBADataset
from stgcnn.model import Social_STGCNN
from stgcnn.loss import bivariate_loss
from utils.metrics import compute_ade, compute_fde, compute_mse
import time

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    print("\n--- Starting Training ---")
    total_batches = len(train_loader)
    
    best_val_mse = float('inf')
    best_val_ade = float('inf')
    checkpoint_path = str(Path(__file__).resolve().parent / "best_model.pt")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        for batch_idx, (X, Y, A) in enumerate(train_loader):
            batch_start = time.time()
            X, Y, A = X.to(device), Y.to(device), A.to(device)
            
            optimizer.zero_grad()
            pred = model(X, A)
            loss = bivariate_loss(pred, Y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                batch_time = time.time() - batch_start
                speed = X.shape[0] / batch_time if batch_time > 0 else 0
                print(f"  Batch {batch_idx:04d}/{total_batches:04d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Speed: {speed:.1f} seq/s | "
                      f"Elapsed: {elapsed:.1f}s")
            
        avg_train_loss = total_train_loss / total_batches
        print(f"-> Epoch {epoch+1} Complete | Average Train NLL: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_ade = 0
        total_fde = 0
        total_mse = 0
        
        register_mu = torch.tensor([0.43076536, 0.04198149]).to(device)
        register_sigma = torch.tensor([29.67073631, 11.54754257]).to(device)
        
        print("  Running Validation...")
        with torch.no_grad():
            for X, Y, A in val_loader:
                X, Y, A = X.to(device), Y.to(device), A.to(device)
                
                mu_x, mu_y, sig_x, sig_y, rho = model(X, A)
                pred = (mu_x, mu_y, sig_x, sig_y, rho)
                
                loss = bivariate_loss(pred, Y)
                total_val_loss += loss.item()
                
                mu = torch.stack([mu_x, mu_y], dim=-1)
                
                mu_denorm = mu * register_sigma + register_mu
                Y_denorm = Y * register_sigma + register_mu
                
                mu_reshaped = mu_denorm.permute(1, 0, 2, 3).reshape(12, -1, 2)
                Y_reshaped = Y_denorm.permute(1, 0, 2, 3).reshape(12, -1, 2)
                
                ade = compute_ade(mu_reshaped, Y_reshaped)
                fde = compute_fde(mu_reshaped, Y_reshaped)
                mse = compute_mse(mu_reshaped, Y_reshaped)
                
                total_ade += ade.item()
                total_fde += fde.item()
                total_mse += mse.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_ade = total_ade / len(val_loader)
        avg_val_fde = total_fde / len(val_loader)
        avg_val_mse = total_mse / len(val_loader)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"-> Epoch {epoch+1:02d} Results | Val NLL: {avg_val_loss:.4f} | "
              f"Val ADE: {avg_val_ade:.4f} ft | Val FDE: {avg_val_fde:.4f} ft | "
              f"Val MSE: {avg_val_mse:.4f} ft^2 | LR: {current_lr:.6f}")
        
        # Checkpoint
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_val_ade = avg_val_ade
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': best_val_ade,
                'val_mse': best_val_mse,
                'val_nll': avg_val_loss,
            }, checkpoint_path)
            print(f"  [Checkpoint] New best Val MSE. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30

    print("Loading datasets...")

    train_dataset = NBADataset(split_file="splits/fold0.json", data_dir="data/train/train", split_key="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = NBADataset(split_file="splits/fold0.json", data_dir="data/train/train", split_key="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Social_STGCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model, train_loader, val_loader, optimizer, scheduler, EPOCHS, device)