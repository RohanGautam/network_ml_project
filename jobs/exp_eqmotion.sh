#!/bin/bash

#SBATCH --job-name=exp_eqm
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ── Experiment matrix ─────────────────────────────────────────────────────────
# Each line is one run: a name followed by the flags forwarded to the entrypoint.
# All runs use the cosine LR scheduler + tuned-best hparams (defaults in the
# script); they differ in normalization and capacity to isolate the effect of
# restoring EqMotion's rotation equivariance via isotropic normalization.
# Each run submits from its OWN best checkpoint (the __main__ --submit path loads
# the checkpointed minimum, not the final-epoch model). Pick the best CSV by the
# printed best val/mse_ft.
COMMON="--max-epochs 300 --patience 40 --full-val"
declare -a RUNS=(
  "aniso_sched_fv   $COMMON"
  "iso_sched_fv     $COMMON --iso-norm"
  "iso_big_fv       $COMMON --iso-norm --hidden-nf 128 --hid-channel 64 --n-layers 4"
)
# ──────────────────────────────────────────────────────────────────────────────

SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

for run in "${RUNS[@]}"; do
    set -- $run
    name="$1"; shift
    echo "==================== RUN: $name ($(date)) ===================="
    python src/equivariance/eqmotion_nba.py --run-name "$name" "$@"
    # Preserve checkpoints AND submissions in $HOME after each run (scratch is
    # wiped by the next job's rsync --delete).
    rsync -a $SCRATCH/network_ml_project/checkpoints/eqmotion/ \
             $HOME/network_ml_project/checkpoints/eqmotion/ 2>/dev/null \
        && echo "Copied checkpoints to HOME"
    rsync -a $SCRATCH/network_ml_project/submissions/ \
             $HOME/network_ml_project/submissions/ 2>/dev/null \
        && echo "Copied submissions to HOME"
done

echo "==================== ALL RUNS DONE ($(date)) ===================="
