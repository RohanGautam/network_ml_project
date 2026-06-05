#!/bin/bash

#SBATCH --job-name=corr_maug
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Cross-arch correlation/ensemble check: aug-MART (val 3.11 / Kaggle 3.01) vs the
# EqMotion ensemble (cached in cache/eqm_residual). Step 1 caches aug-MART's K=20
# predictions on the SAME 3078 deterministic val windows; step 2 runs the
# correlation + weighted-ensemble + closed-form-optimal-weight analysis.
ROOT=$HOME/network_ml_project
AUGCKPT="src/mart/checkpoints/mart_aug_iso_hoops_5k_best.ckpt"

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $ROOT $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART imports require src/mart as cwd
source $ROOT/.venv/bin/activate

echo "==================== STEP 1: cache aug-MART val preds ===================="
python cache_mart_preds.py \
    --checkpoint "$SCRATCH/network_ml_project/$AUGCKPT" \
    --split_path "$SCRATCH/network_ml_project/splits/fold0.json" \
    --cache_dir "$SCRATCH/network_ml_project/cache/mart_aug_5k"

# Persist the cache back to HOME (scratch is wiped on the next job's rsync).
mkdir -p $ROOT/cache/mart_aug_5k
rsync -a "$SCRATCH/network_ml_project/cache/mart_aug_5k/" $ROOT/cache/mart_aug_5k/ \
    && echo "Copied aug-MART cache to HOME"

echo "==================== STEP 2: correlation + ensemble analysis ===================="
python corr_augmart_eqmotion.py \
    --augmart_cache "$SCRATCH/network_ml_project/cache/mart_aug_5k" \
    --eqm_cache "$SCRATCH/network_ml_project/cache/eqm_residual"

echo "==================== DONE ($(date)) ===================="
