#!/bin/bash

#SBATCH --job-name=ball_probe
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Possession-anchored ball diagnostic on the best aug-MART checkpoint. Zero
# training. Tests whether anchoring the ball to its likely handler beats the
# free-agent model (the structural lever for the ball-dominated 3.1->2.6 gap).
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
trap "rm -rf $SCRATCH" EXIT
cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

python ball_possession_probe.py \
    --checkpoint "$SCRATCH/network_ml_project/src/mart/checkpoints/mart_aug_iso_hoops_5k_best.ckpt" \
    --split_path "$SCRATCH/network_ml_project/splits/fold0.json"
echo "==================== DONE ($(date)) ===================="
