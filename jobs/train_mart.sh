#!/bin/bash

#SBATCH --job-name=tr_m
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Two MART runs, sequential in one job: (1) the paper-native min-of-K ADE loss
# (MART's default) and (2) the mean-MSE loss aligned with EqMotion / Kaggle's
# single-shot scoring. Both share the same epoch budget + cosine LR schedule
# (set in configs/mart_nba_pt.yaml) so the only difference is the loss.
# Extra flags after the script are forwarded to BOTH runs (e.g. --use_hoops).
declare -a RUNS=(
  "mart_minade_s1     "
  "mart_meanmse_s1    --loss mean_mse"
)

# Preserve any flags passed to sbatch (e.g. --use_hoops) so we can forward them
# to every run; the inner `set -- $run` below overwrites $@ with per-run args.
EXTRA=("$@")

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART's imports require src/mart as cwd
source $HOME/network_ml_project/.venv/bin/activate

mkdir -p $HOME/network_ml_project/src/mart/checkpoints

for run in "${RUNS[@]}"; do
    set -- $run
    name="$1"; shift
    echo "==================== RUN: $name ($(date)) ===================="
    python main_nba_pt.py \
        --config configs/mart_nba_pt.yaml \
        --split_path $SCRATCH/network_ml_project/splits/fold0.json \
        --model_name "$name" \
        --wandb_run_name "$name" \
        "$@" "${EXTRA[@]}"
    # Preserve checkpoints after each run (scratch is wiped on next job's rsync).
    rsync -a $SCRATCH/network_ml_project/src/mart/checkpoints/ \
             $HOME/network_ml_project/src/mart/checkpoints/ 2>/dev/null \
        && echo "Copied MART checkpoints to HOME"
done

echo "==================== ALL RUNS DONE ($(date)) ===================="
