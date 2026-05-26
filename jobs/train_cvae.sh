#!/bin/bash

#SBATCH --job-name=train_cvae
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1


SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"


cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate


python src/stgcnn/stgcnn_cvae.py

# Preserve the best checkpoint in HOME (scratch is wiped by the next job's rsync).
mkdir -p $HOME/network_ml_project/checkpoints/cvae
cp -f $SCRATCH/network_ml_project/checkpoints/cvae/best.ckpt \
      $HOME/network_ml_project/checkpoints/cvae/best.ckpt 2>/dev/null \
  && echo "Saved checkpoint to HOME"
