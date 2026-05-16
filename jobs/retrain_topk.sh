#!/bin/bash

#SBATCH --job-name=topk_eqm
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

# Optuna study lives in $HOME (persistent); pass the absolute path so the script
# reads it directly rather than the rsynced copy.
STUDY_PATH="$HOME/network_ml_project/optuna_studies/eqmotion.db"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

python src/equivariance/retrain_topk.py \
    --study-path "$STUDY_PATH" \
    --study-name eqmotion_v1 \
    --top-k 5 \
    --max-epochs 200 \
    --patience 20

echo "TRAINING DONE AT $(date)"

# Copy submissions and topk checkpoints back to $HOME so they survive /scratch.
mkdir -p $HOME/network_ml_project/submissions $HOME/network_ml_project/models/topk
rsync -a $SCRATCH/network_ml_project/submissions/ $HOME/network_ml_project/submissions/
rsync -a $SCRATCH/network_ml_project/models/topk/ $HOME/network_ml_project/models/topk/
echo "SYNCED BACK AT $(date)"
