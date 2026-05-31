#!/bin/bash

#SBATCH --job-name=mart_res
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Train+submit-only variant of the residual MART pipeline: reuses the
# EqMotion-ensemble cache from jobs/train_mart_residual.sh (cache lives in
# $HOME/cache/eqm_residual). Use this to iterate the loss / hparams without
# re-running EqMotion's 5-seed ensemble each time.
#
# Usage:
#   sbatch jobs/train_mart_residual_only.sh --tag mart_residual_v2 --loss_kind mean_mse
TAG="mart_residual_v2"
REST=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        *) REST+=("$1"); shift ;;
    esac
done

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

CACHE_DIR=$SCRATCH/network_ml_project/cache/eqm_residual
mkdir -p "$CACHE_DIR"
rsync -a "$HOME/network_ml_project/cache/eqm_residual/" "$CACHE_DIR/"
echo "Cache files in $CACHE_DIR:"
ls -la "$CACHE_DIR"

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

mkdir -p $HOME/network_ml_project/src/mart/checkpoints
OUT_CKPT="$HOME/network_ml_project/src/mart/checkpoints/${TAG}.pt"

echo "==================== STAGE 2: train MART residual ($TAG) ===================="
python train_mart_residual.py \
    --cache_dir "$CACHE_DIR" \
    --mart_config configs/mart_nba_pt.yaml \
    --out_ckpt "$OUT_CKPT" \
    --epochs 150 \
    "${REST[@]}"

echo "==================== STAGE 3: write submission ===================="
TS=$(date +%Y%m%d_%H%M%S)
OUT_CSV="$HOME/network_ml_project/submissions/solution_${TAG}_${TS}.csv"
mkdir -p "$(dirname "$OUT_CSV")"

python submit_mart_residual.py \
    --cache_dir "$CACHE_DIR" \
    --mart_residual_ckpt "$OUT_CKPT" \
    --out_csv "$OUT_CSV"

echo "==================== DONE ($(date)) — wrote $OUT_CSV ===================="
