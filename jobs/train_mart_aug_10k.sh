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

# Per-job scratch dir keyed on the Slurm job id. CRITICAL: concurrent jobs must
# NOT share one scratch tree — the `rsync --delete` below would otherwise wipe a
# sibling job's in-progress run (this exact collision killed SGDR job 2955010
# when the 10k job started and deleted its scratch mid-run). Unique per job =>
# safe to run many jobs in parallel.
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH"
# Clean up this job's scratch copy on exit (scratch is a shared, quota'd FS).
trap "rm -rf $SCRATCH" EXIT

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
