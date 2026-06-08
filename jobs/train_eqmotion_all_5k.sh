#!/bin/bash

#SBATCH --job-name=eqm_all5k
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

# FINAL Kaggle model: best-EqMotion recipe (iso norm + hoops + cosine + tuned
# hparams) trained for 5000 epochs on the FULL labelled dataset (train+val folded
# together via --train-all) — no held-out fold, no val/checkpoint selection, no
# early stopping. Submits the final-epoch model. Use the eqm_iso_hoops_5k split
# run (same recipe, with val) as the trustworthy val estimate for this model.
NAME="eqm_iso_hoops_all_5k"

# Per-job scratch so concurrent runs don't collide on the shared-scratch
# rsync --delete (which would wipe the other job's working tree mid-training).
SCRATCH="/scratch/izar/rgautam/job_$SLURM_JOB_ID"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

echo "==================== RUN: $NAME ($(date)) ===================="
python src/equivariance/eqmotion_nba.py \
    --run-name "$NAME" \
    --iso-norm --add-hoops \
    --train-all \
    --max-epochs 5000

# Preserve checkpoints AND submissions in $HOME (scratch is wiped by the next
# job's rsync --delete).
rsync -a $SCRATCH/network_ml_project/checkpoints/eqmotion/ \
         $HOME/network_ml_project/checkpoints/eqmotion/ 2>/dev/null \
    && echo "Copied checkpoints to HOME"
rsync -a $SCRATCH/network_ml_project/submissions/ \
         $HOME/network_ml_project/submissions/ 2>/dev/null \
    && echo "Copied submissions to HOME"

echo "==================== DONE ($(date)) ===================="
