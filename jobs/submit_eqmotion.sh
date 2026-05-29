#!/bin/bash

#SBATCH --job-name=submit_eqm
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Args after the script are forwarded, e.g.:
#   sbatch jobs/submit_eqmotion.sh --ckpt checkpoints/eqmotion/iso_sched/best.ckpt --iso-norm

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

python src/equivariance/submit_eqmotion.py "$@"

# Preserve the generated CSV in $HOME (scratch is wiped by the next job's rsync).
rsync -a $SCRATCH/network_ml_project/submissions/ \
         $HOME/network_ml_project/submissions/ 2>/dev/null \
    && echo "Copied submissions to HOME"
