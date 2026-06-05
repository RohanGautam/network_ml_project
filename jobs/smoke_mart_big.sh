#!/bin/bash

#SBATCH --job-name=smk_big
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# 3-epoch smoke of the scaled-up big config: confirm it builds, fits GPU memory,
# report param count + per-epoch wall (to size the 5k run vs the 24h limit).
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
trap "rm -rf $SCRATCH" EXIT
cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

python main_nba_pt.py \
    --config configs/mart_nba_aug_big.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name smoke_big --wandb_mode disabled \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 3
echo "==================== SMOKE DONE ($(date)) ===================="
