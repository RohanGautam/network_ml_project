#!/bin/bash

#SBATCH --job-name=smk_mmse
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

# 3-epoch smoke of --loss mean_mse with the full aug regime: confirm the flag
# runs end-to-end (aug + hoops + iso) and that train_loss is a plain MSE (small,
# decreasing) before committing the 5000-ep run.
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
trap "rm -rf $SCRATCH" EXIT
cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name smoke_mmse --wandb_mode disabled \
    --loss mean_mse \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 3
echo "==================== SMOKE DONE ($(date)) ===================="
