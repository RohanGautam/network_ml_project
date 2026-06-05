#!/bin/bash

#SBATCH --job-name=tr_mmse
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# OBJECTIVE-MATCH run: identical to the proven 7.5M aug-MART (iso + O(2) aug +
# hoops, 5000-ep cosine) EXCEPT --loss mean_mse instead of min_ade.
#
# Why: the agg_probe showed the min_ade model's mean-of-K (3.11, what we submit)
# is a BIASED estimate of the conditional mean E[y|x] -- best-of-K training lets
# the sample cloud's centre drift ~2.5 ft off truth (oracle best-sample = 0.45,
# but it's in the low-density tail so unsupervised selection can't reach it).
# mean_mse = mean_k MSE(s_k,gt) = MSE(mean_k,gt) + spread, so it drives the
# prediction straight to E[y|x] (the MMSE point that minimises our submitted
# metric) and removes the best-of-K bias. Direct test of whether the conditional
# mean is sharpenable below 3.11 (leaderboard 2.6 implies it is).
NAME="mart_meanmse_iso_hoops_5k"

SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH"
trap "rm -rf $SCRATCH" EXIT

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate
mkdir -p $HOME/network_ml_project/src/mart/checkpoints

echo "==================== RUN: $NAME ($(date)) ===================="
python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name "$NAME" --wandb_run_name "$NAME" \
    --loss mean_mse \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 5000

rsync -a $SCRATCH/network_ml_project/src/mart/checkpoints/ \
         $HOME/network_ml_project/src/mart/checkpoints/ 2>/dev/null \
    && echo "Copied checkpoints to HOME"
echo "==================== DONE ($(date)) ===================="
