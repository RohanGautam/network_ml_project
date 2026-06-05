#!/bin/bash

#SBATCH --job-name=tr_mseed
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

# One aug-MART seed for a multi-seed ensemble. Identical recipe to the 3.11
# backbone (iso + full O(2) aug + hoops, 5000 ep) but a fresh init/seed, so the
# resulting models decorrelate MORE than SGDR cycle minima (which share one
# trajectory). Pass the seed as the first arg: sbatch train_mart_aug_seed.sh 2
SEED="${1:-2}"
NAME="mart_aug_iso_hoops_s${SEED}"

SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH  (seed=$SEED)"
trap "rm -rf $SCRATCH" EXIT

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate
mkdir -p $HOME/network_ml_project/src/mart/checkpoints

echo "==================== RUN: $NAME ($(date)) ===================="
python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path $SCRATCH/network_ml_project/splits/fold0.json \
    --model_name "$NAME" --wandb_run_name "$NAME" --seed "$SEED" \
    --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror \
    --num_epochs 5000

rsync -a $SCRATCH/network_ml_project/src/mart/checkpoints/ \
         $HOME/network_ml_project/src/mart/checkpoints/ 2>/dev/null \
    && echo "Copied checkpoints to HOME"
echo "==================== DONE ($(date)) ===================="
