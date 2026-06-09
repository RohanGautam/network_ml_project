#!/bin/bash

#SBATCH --job-name=eqm_5k
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

# Best-EqMotion recipe (iso norm + hoops + cosine + tuned-best hparams), but run
# for 5000 epochs on the held-out train/val split — the long-schedule analogue of
# the aug-MART 5k run, to see if EqMotion keeps descending past its 300-ep floor.
# Keeps the val fold so val/mse_ft stays a trustworthy selector; submits from the
# best-by-val checkpoint. Patience is set above max-epochs to disable early
# stopping so the full cosine schedule runs.
NAME="eqm_iso_hoops_5k"

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
    --iso-norm --add-hoops --full-val \
    --max-epochs 5000 --patience 6000

# Preserve checkpoints AND submissions in $HOME (scratch is wiped by the next
# job's rsync --delete).
rsync -a $SCRATCH/network_ml_project/checkpoints/eqmotion/ \
         $HOME/network_ml_project/checkpoints/eqmotion/ 2>/dev/null \
    && echo "Copied checkpoints to HOME"
rsync -a $SCRATCH/network_ml_project/submissions/ \
         $HOME/network_ml_project/submissions/ 2>/dev/null \
    && echo "Copied submissions to HOME"

echo "==================== DONE ($(date)) ===================="
