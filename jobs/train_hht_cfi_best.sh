#!/bin/bash

#SBATCH --job-name=hht_best
#SBATCH --output=$HOME/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=$HOME/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=ee-452

SCRATCH="/scratch/izar/$USER/"
rsync -a --delete --exclude='__pycache__' --exclude='*.db' --exclude='logs/' \
      $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nanovlm

python src/hht_cfi/train_best.py

# Sync logs (checkpoints) back to home
rsync -a logs/ "$HOME/network_ml_project/logs/" 2>/dev/null && echo "Logs synced back."

echo "DONE AT $(date)"
