#!/bin/bash

#SBATCH --job-name=sel_aug5k
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Learned head selector on the BEST model (mart_aug_iso_hoops_5k) -- untried.
# Prior selector was on the 300ep/no-aug mart_minade_s1 (ball 16.78, ~5% gain,
# overfit). The aug-5k model (ball 13.5, heavy O(2) aug, 5000ep) should have
# cleaner/more-distinct modes -> more selectable. Math: capturing ~50% of the
# ball oracle headroom -> total ~2.6. Combat the prior overfit with stronger
# regularization (higher weight_decay + dropout) and a few variants.
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
trap "rm -rf $SCRATCH" EXIT
cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

CKPT="$SCRATCH/network_ml_project/src/mart/checkpoints/mart_aug_iso_hoops_5k_best.ckpt"
CACHE="$SCRATCH/network_ml_project/cache/mart_aug_iso_hoops_5k"
SPLIT="$SCRATCH/network_ml_project/splits/fold0.json"

echo "==================== CACHE aug-5k K=20 preds ($(date)) ===================="
python cache_mart_preds.py --checkpoint "$CKPT" --cache_dir "$CACHE" --split_path "$SPLIT"

OUT=$HOME/network_ml_project/src/mart/checkpoints
mkdir -p $OUT
for TAG in "v1:--hidden 128 --dropout 0.2 --weight_decay 1e-3" \
           "v2_ballonly:--hidden 128 --dropout 0.2 --weight_decay 1e-3 --ball_only_loss" \
           "v3_noscene:--hidden 128 --dropout 0.3 --weight_decay 3e-3 --no_scene_context"; do
    NAME="${TAG%%:*}"; ARGS="${TAG#*:}"
    echo "==================== SELECTOR $NAME  [$ARGS] ($(date)) ===================="
    python train_head_selector.py --cache_dir "$CACHE" \
        --out_ckpt "$OUT/head_selector_aug5k_${NAME}.pt" \
        --epochs 120 --monitor ball $ARGS
done
echo "==================== DONE ($(date)) ===================="
