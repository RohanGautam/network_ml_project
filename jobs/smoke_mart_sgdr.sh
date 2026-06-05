#!/bin/bash

#SBATCH --job-name=smk_sgdr
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Quick SGDR-path validation: 6 epochs, T_0=2 T_mult=1 -> 3 restarts -> 3 cycle
# snapshots. Confirms the CosineAnnealingWarmRestarts branch builds, restarts are
# detected (LR jump), and per-cycle snapshots save. NOT a result run.
SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"
cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name smoke_sgdr --wandb_mode disabled \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 6 \
    --scheduler_type CosineAnnealingWarmRestarts --sgdr_t0 2 --sgdr_tmult 1

echo "=== snapshots ==="; ls -la checkpoints/smoke_sgdr_snapshots/ 2>/dev/null
echo "==================== SMOKE DONE ($(date)) ===================="
