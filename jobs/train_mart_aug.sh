#!/bin/bash

#SBATCH --job-name=tr_maug
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

# "Augment + scale + train-long" MART experiment. Gives a NON-equivariant MART
# the same inductive structure that won for EqMotion (isotropic norm + full
# rotation/reflection invariance + hoop nodes), but via augmentation instead of
# hard equivariance — so it can actually scale capacity / train long without
# overfitting. Uses the bigger configs/mart_nba_aug.yaml (1000 epochs, cosine).
#
# Two variants, sequential in one job:
#   1. iso + full O(2) aug + hoops  (full EqMotion-recipe transfer)
#   2. iso + full O(2) aug, NO hoops (ablation: does aug alone match hoops here?)
# Loss stays min_ade (experiments.md: the K=20 heads act as an implicit ensemble;
# mean_mse collapses that diversity and scored worse).
declare -a RUNS=(
  "mart_aug_iso_hoops   --use_hoops --iso_norm --aug_rot_deg 180 --aug_court_mirror"
  "mart_aug_iso_nohoops             --iso_norm --aug_rot_deg 180 --aug_court_mirror"
)

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
        --config configs/mart_nba_aug.yaml \
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
