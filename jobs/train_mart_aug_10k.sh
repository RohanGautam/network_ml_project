#!/bin/bash

#SBATCH --job-name=tr_m10k
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Single variant: iso + full O(2) aug + hoops, 10000 epochs, SINGLE cosine
# (T_max=10000 set via --num_epochs; no code change). Tests whether yet more
# single-cosine compute beats the 5000-epoch 3.11. Counterpoint to the SGDR run:
#   5k cosine (3.11)  vs  10k cosine (this)  vs  5k SGDR
# isolates "more compute" from "restart shape". New model_name so the 5k and
# SGDR checkpoints are preserved.
NAME="mart_aug_iso_hoops_10k"

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART imports require src/mart as cwd
source $HOME/network_ml_project/.venv/bin/activate

mkdir -p $HOME/network_ml_project/src/mart/checkpoints

echo "==================== RUN: $NAME ($(date)) ===================="
python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name "$NAME" \
    --wandb_run_name "$NAME" \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 10000

# Preserve checkpoints (scratch is wiped on the next job's rsync --delete).
rsync -a $SCRATCH/network_ml_project/src/mart/checkpoints/ \
         $HOME/network_ml_project/src/mart/checkpoints/ 2>/dev/null \
    && echo "Copied MART checkpoints to HOME"

echo "==================== DONE ($(date)) ===================="
