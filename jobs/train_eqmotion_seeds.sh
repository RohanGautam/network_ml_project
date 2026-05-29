#!/bin/bash

#SBATCH --job-name=eqm_seeds
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Train N seeds of the best EqMotion config (iso norm + hoops + cosine, honest
# 11-entity val metric) for a multi-seed ensemble. No per-run submission — the
# ensemble script averages checkpoints and submits once.
NSEEDS=${1:-5}
COMMON="--iso-norm --add-hoops --full-val --max-epochs 300 --patience 40 --no-submit"

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

for s in $(seq 0 $((NSEEDS-1))); do
    echo "==================== SEED $s ($(date)) ===================="
    python src/equivariance/eqmotion_nba.py \
        --run-name "iso_hoops_s${s}" --seed "$s" $COMMON
    rsync -a $SCRATCH/network_ml_project/checkpoints/eqmotion/ \
             $HOME/network_ml_project/checkpoints/eqmotion/ 2>/dev/null \
        && echo "Copied checkpoints to HOME"
done

echo "==================== ALL SEEDS DONE ($(date)) ===================="
