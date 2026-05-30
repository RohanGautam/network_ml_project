#!/bin/bash

#SBATCH --job-name=head_sel
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Two-stage pipeline:
#   1) Cache MART's K=20 predictions on train+val (one-time MART forward pass).
#   2) Train a per-agent head selector on those cached predictions.
# Selector is small and trains in minutes once the cache exists.
MART_CKPT="src/mart/checkpoints/mart_minade_s1.ckpt"
CACHE_NAME="mart_minade_s1"
SELECTOR_TAG="head_selector_v1"

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

CACHE_DIR=$SCRATCH/network_ml_project/cache/$CACHE_NAME
mkdir -p "$CACHE_DIR"
mkdir -p $HOME/network_ml_project/cache/$CACHE_NAME
mkdir -p $HOME/network_ml_project/src/mart/checkpoints

echo "==================== STAGE 1: cache MART predictions ===================="
python cache_mart_preds.py \
    --checkpoint "$SCRATCH/network_ml_project/$MART_CKPT" \
    --split_path "$SCRATCH/network_ml_project/splits/fold0.json" \
    --cache_dir "$CACHE_DIR" \
    --windows_per_seq 8

# Preserve cache in $HOME so we can iterate the selector without re-running MART.
rsync -a "$CACHE_DIR/" "$HOME/network_ml_project/cache/$CACHE_NAME/"

echo "==================== STAGE 2: train head selector ===================="
python train_head_selector.py \
    --cache_dir "$CACHE_DIR" \
    --out_ckpt "$HOME/network_ml_project/src/mart/checkpoints/${SELECTOR_TAG}.pt" \
    --epochs 80 \
    --hidden 128 \
    --lr 1e-3 \
    --monitor ball

echo "==================== DONE ($(date)) ===================="
