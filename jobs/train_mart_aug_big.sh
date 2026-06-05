#!/bin/bash

#SBATCH --job-name=tr_mbig
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

# Capacity-ceiling test: ~3-4x bigger aug-MART (mart_nba_aug_big.yaml, 256-512-8)
# at the SAME 5000-ep cosine + iso + full O(2) aug + hoops. Tests whether the
# ~3.1 shared error floor is a capacity limit (helps) or a data floor (train≈val
# persists ~3.1). Watch train-vs-val: if train collapses below val => overfit
# (capacity now binds); if they stay equal => data floor confirmed.
NAME="mart_aug_iso_hoops_big5k"

SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH"
trap "rm -rf $SCRATCH" EXIT

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate
mkdir -p $HOME/network_ml_project/src/mart/checkpoints

echo "==================== RUN: $NAME ($(date)) ===================="
python main_nba_pt.py \
    --config configs/mart_nba_aug_big.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name "$NAME" --wandb_run_name "$NAME" \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 5000

rsync -a $SCRATCH/network_ml_project/src/mart/checkpoints/ \
         $HOME/network_ml_project/src/mart/checkpoints/ 2>/dev/null \
    && echo "Copied checkpoints to HOME"
echo "==================== DONE ($(date)) ===================="
