#!/bin/bash

#SBATCH --job-name=agg_probe
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Mode-aggregation headroom probe: measure sample diversity and compare
# mean / medoid / densest-cluster / per-entity-oracle aggregations on the
# existing (collapsed-mode?) 5k and 10k checkpoints. Zero training.
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH"
trap "rm -rf $SCRATCH" EXIT

cd $SCRATCH/network_ml_project/src/mart
source $HOME/network_ml_project/.venv/bin/activate

for CK in mart_aug_iso_hoops_5k_best mart_aug_iso_hoops_10k_best; do
    echo "==================== $CK ($(date)) ===================="
    python agg_probe.py \
        --checkpoint "$SCRATCH/network_ml_project/src/mart/checkpoints/$CK.ckpt" \
        --split_path "$SCRATCH/network_ml_project/splits/fold0.json"
done
echo "==================== DONE ($(date)) ===================="
