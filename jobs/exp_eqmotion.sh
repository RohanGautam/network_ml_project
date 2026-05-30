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
  # Court-landmark sweep. Hoops alone gave honest val 3.33 / Kaggle 3.2 — the
  # single biggest lever found. Try richer D2-respecting landmark sets to see
  # whether more court structure pushes further. Each preset is a distinct
  # team_id so the model's id_embed can specialize per landmark type. Control:
  # iso_hoops_s0=3.33 (same seed, same code path, hoops only).
  "iso_l_hoops_ft        $COMMON --iso-norm --landmarks hoops,ft"
  "iso_l_hoops_3pt       $COMMON --iso-norm --landmarks hoops,3pt"
  "iso_l_hoops_corners   $COMMON --iso-norm --landmarks hoops,corners"
  "iso_l_hoops_ft_3pt    $COMMON --iso-norm --landmarks hoops,ft,3pt"
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
