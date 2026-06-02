#!/bin/bash

#SBATCH --job-name=tr_msgdr
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

# SGDR (cosine warm restarts) — controlled test of the schedule-SHAPE hypothesis.
# BUDGET-MATCHED to the single-cosine 5k baseline (same 5000 epochs, same 7.5M
# aug config, same iso + full O(2) aug + hoops, same eta_min=lr*0.02), so any
# delta vs the 3.11 single-cosine result is attributable to the LR trajectory
# (periodic restarts) and NOT to extra compute.
#
# 10 equal cycles of 500 epochs (T_0=500, T_mult=1). Saves one snapshot per cycle
# minimum -> a free same-arch snapshot ensemble (a separate lever from the
# schedule question; captured regardless of whether any single cycle beats 3.11).
NAME="mart_aug_iso_hoops_sgdr"

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
    --num_epochs 5000 \
    --scheduler_type CosineAnnealingWarmRestarts \
    --sgdr_t0 500 --sgdr_tmult 1

# Preserve checkpoints + per-cycle snapshots (scratch wiped on next job's rsync).
rsync -a $SCRATCH/network_ml_project/src/mart/checkpoints/ \
         $HOME/network_ml_project/src/mart/checkpoints/ 2>/dev/null \
    && echo "Copied MART checkpoints + snapshots to HOME"

echo "==================== DONE ($(date)) ===================="
