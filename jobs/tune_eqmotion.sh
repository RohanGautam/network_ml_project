#!/bin/bash

#SBATCH --job-name=tune_eqm
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

# Persistent Optuna study lives in $HOME so it survives /scratch wipes and resumes
# across re-launches of this job.
STUDY_DIR="$HOME/network_ml_project/optuna_studies"
mkdir -p "$STUDY_DIR"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

# Stop ~30 min before SLURM kills us so the last trial can finish & DB can flush.
TIMEOUT=$(( 12*3600 - 1800 ))

python src/equivariance/tune_eqmotion.py \
    --study-name eqmotion_v1 \
    --study-path "$STUDY_DIR/eqmotion.db" \
    --n-trials 60 \
    --max-epochs 50 \
    --patience 10 \
    --timeout "$TIMEOUT"
