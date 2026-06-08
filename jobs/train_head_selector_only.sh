#!/bin/bash

#SBATCH --job-name=head_sel
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Train-only variant of the head selector: assumes cache_mart_preds.py has
# already been run (cache lives in $HOME/cache/<name>). Iterates the selector
# architecture/hparams without re-running MART each time. Forwards any args
# after the script to train_head_selector.py.
#
# Usage (default v2 = scene context):
#   sbatch jobs/train_head_selector_only.sh --tag head_selector_v2 --hidden 256
CACHE_NAME="mart_minade_s1"

# Light arg parsing for --tag (controls output filename) so the script is
# self-documenting; all other flags forward to train_head_selector.py as-is.
TAG="head_selector_v2"
REST=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        *) REST+=("$1"); shift ;;
    esac
done

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

# Pull the cache from HOME into the expected scratch path so the trainer reads
# the same path on every cluster node.
CACHE_DIR=$SCRATCH/network_ml_project/cache/$CACHE_NAME
mkdir -p "$CACHE_DIR"
rsync -a "$HOME/network_ml_project/cache/$CACHE_NAME/" "$CACHE_DIR/"
echo "Cache files in $CACHE_DIR:"
ls -la "$CACHE_DIR"

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

mkdir -p $HOME/network_ml_project/src/mart/checkpoints
OUT_CKPT="$HOME/network_ml_project/src/mart/checkpoints/${TAG}.pt"

python train_head_selector.py \
    --cache_dir "$CACHE_DIR" \
    --out_ckpt "$OUT_CKPT" \
    "${REST[@]}"

echo "==================== DONE ($(date)) ===================="
