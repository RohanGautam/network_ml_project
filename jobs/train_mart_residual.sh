#!/bin/bash

#SBATCH --job-name=mart_res
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Boosting-style pipeline:
#   1. Cache EqMotion 5-seed ensemble predictions on train+val+test.
#   2. Train fresh MART to predict EqMotion's PLAYER residuals (ball gradient
#      zeroed since its residual is irreducibly multimodal per the head-selector
#      sweep).
#   3. Combine EqMotion (base) + MART (player residual) into a Kaggle CSV.
# All three stages run sequentially in one job; cache + residual checkpoint are
# rsynced to $HOME after each stage so we can iterate stage 2/3 on cheap reads.

TAG="mart_residual_v1"

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART imports require src/mart as cwd
source $HOME/network_ml_project/.venv/bin/activate

CACHE_DIR=$SCRATCH/network_ml_project/cache/eqm_residual
mkdir -p "$CACHE_DIR"
mkdir -p $HOME/network_ml_project/cache/eqm_residual
mkdir -p $HOME/network_ml_project/src/mart/checkpoints

echo "==================== STAGE 1: cache EqMotion ensemble ===================="
python cache_eqmotion_residuals.py \
    --eqmotion_ckpts "$SCRATCH/network_ml_project/checkpoints/eqmotion/iso_hoops_s*/best.ckpt" \
    --mart_ckpt "$SCRATCH/network_ml_project/src/mart/checkpoints/mart_minade_s1.ckpt" \
    --split_path "$SCRATCH/network_ml_project/splits/fold0.json" \
    --test_dir "$SCRATCH/network_ml_project/data/test/test" \
    --cache_dir "$CACHE_DIR" \
    --windows_per_seq 8

rsync -a "$CACHE_DIR/" "$HOME/network_ml_project/cache/eqm_residual/"

echo "==================== STAGE 2: train MART residual ===================="
python train_mart_residual.py \
    --cache_dir "$CACHE_DIR" \
    --mart_config configs/mart_nba_pt.yaml \
    --out_ckpt "$HOME/network_ml_project/src/mart/checkpoints/${TAG}.pt" \
    --epochs 150

echo "==================== STAGE 3: write submission ===================="
TS=$(date +%Y%m%d_%H%M%S)
OUT_CSV="$HOME/network_ml_project/submissions/solution_${TAG}_${TS}.csv"
mkdir -p "$(dirname "$OUT_CSV")"

python submit_mart_residual.py \
    --cache_dir "$CACHE_DIR" \
    --mart_residual_ckpt "$HOME/network_ml_project/src/mart/checkpoints/${TAG}.pt" \
    --out_csv "$OUT_CSV"

echo "==================== DONE ($(date)) — wrote $OUT_CSV ===================="
