#!/bin/bash

#SBATCH --job-name=snap_ens
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Evaluate the SGDR snapshot ensemble (10 per-cycle minima). Per-job scratch so
# it can't collide with the still-running 10k job.
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH"
trap "rm -rf $SCRATCH" EXIT

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

python snapshot_ensemble_eval.py \
    --snapshot_dir "$SCRATCH/network_ml_project/src/mart/checkpoints/mart_aug_iso_hoops_sgdr_snapshots" \
    --split_path "$SCRATCH/network_ml_project/splits/fold0.json"

echo "==================== DONE ($(date)) ===================="
