#!/bin/bash

#SBATCH --job-name=sub_maug
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Generate the Kaggle submission from the 5000-epoch aug-MART best checkpoint
# (val/mse_ft 3.112 — new single-model best). submit_nba_pt.py reads mu/sigma +
# use_hoops from the checkpoint, so iso-norm denorm is handled automatically.
ROOT=$HOME/network_ml_project
CKPT="$ROOT/src/mart/checkpoints/mart_aug_iso_hoops_5k_best.ckpt"
CSV="$ROOT/submissions/solution_mart_aug_5k_best.csv"

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $ROOT $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART imports require src/mart as cwd
source $ROOT/.venv/bin/activate

# Use the scratch copy of the checkpoint (synced above) to avoid HOME contention.
python submit_nba_pt.py \
    --checkpoint "$SCRATCH/network_ml_project/src/mart/checkpoints/mart_aug_iso_hoops_5k_best.ckpt" \
    --test_dir "$SCRATCH/network_ml_project/data/test/test" \
    --out_csv "$SCRATCH/network_ml_project/submissions/solution_mart_aug_5k_best.csv"

# Copy the submission back to HOME (scratch is wiped on the next job's rsync).
mkdir -p $ROOT/submissions
rsync -a "$SCRATCH/network_ml_project/submissions/solution_mart_aug_5k_best.csv" "$CSV" \
    && echo "Copied submission to $CSV"
echo "--- head ---"; head -2 "$CSV" | cut -c1-100
wc -l "$CSV"
echo "==================== SUBMISSION DONE ($(date)) ===================="
