#!/bin/bash

#SBATCH --job-name=submit_mart
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Generate a Kaggle submission CSV from a MART checkpoint.
# Required:  --checkpoint <path>   (path relative to repo root or absolute)
# Optional:  --reduce mean|first   (K-head reduction; mean is best for MSE)
# Example:
#   sbatch jobs/submit_mart.sh --checkpoint src/mart/checkpoints/mart_baseline_s1.ckpt

# Resolve --checkpoint to an absolute path BEFORE we cd, so the path survives
# the cwd change to src/mart below.
CKPT=""
REST=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CKPT="$(readlink -f "$2")"; shift 2 ;;
        *)
            REST+=("$1"); shift ;;
    esac
done
[[ -z "$CKPT" ]] && { echo "ERROR: --checkpoint <path> is required" >&2; exit 1; }

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart  # MART's imports require src/mart as cwd
source $HOME/network_ml_project/.venv/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
OUT_CSV="$SCRATCH/network_ml_project/submissions/solution_mart_${TS}.csv"
mkdir -p "$(dirname "$OUT_CSV")"

python submit_nba_pt.py \
    --checkpoint "$CKPT" \
    --test_dir $SCRATCH/network_ml_project/data/test/test \
    --out_csv "$OUT_CSV" \
    "${REST[@]}"

# Preserve submission in $HOME.
rsync -a $SCRATCH/network_ml_project/submissions/ \
         $HOME/network_ml_project/submissions/ 2>/dev/null \
    && echo "Copied submissions to HOME"
