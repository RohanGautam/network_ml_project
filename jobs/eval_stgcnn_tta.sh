#!/bin/bash

#SBATCH --job-name=eval_stgcnn_tta
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:30:00
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


python src/stgcnn/stgcnn_nba.py --ckpt checkpoints/7z5clshn_aug.ckpt
