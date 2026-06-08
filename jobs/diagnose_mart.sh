#!/bin/bash

#SBATCH --job-name=diag_mart
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=0:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Per-entity (ball vs players) MSE breakdown on the existing MART checkpoints.
# Tells us whether MART's ball MSE is in a different regime than EqMotion's
# ~13.7 floor — which would mean a MART ball-specialist is worth training.

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART imports require src/mart as cwd
source $HOME/network_ml_project/.venv/bin/activate

python diagnose_per_entity.py \
    --checkpoints \
        checkpoints/mart_minade_s1.ckpt \
        checkpoints/mart_meanmse_s1.ckpt \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json
