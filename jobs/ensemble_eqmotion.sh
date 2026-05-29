#!/bin/bash

#SBATCH --job-name=eqm_ens
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Evaluate {single,ensemble}x{plain,tta} on val and write ensemble submissions.
# Run after train_eqmotion_seeds.sh (use --dependency=afterok:<seed_job_id>).

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

python src/equivariance/ensemble_eqmotion.py \
    --ckpts "checkpoints/eqmotion/iso_hoops_s*/best.ckpt"

rsync -a $SCRATCH/network_ml_project/submissions/ \
         $HOME/network_ml_project/submissions/ 2>/dev/null \
    && echo "Copied submissions to HOME"
