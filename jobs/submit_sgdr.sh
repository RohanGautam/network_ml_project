#!/bin/bash

#SBATCH --job-name=sub_sgdr
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Build SGDR submissions + the best-3 x EqMotion blend, end to end:
#   (1) SGDR best-by-val single model            -> solution_sgdr_best_val3.116.csv
#   (2) best-3 snapshot ensemble (cycles 07,08,09) -> solution_sgdr_snap_best3_val3.101.csv
#   (3) tune best-3 x EqMotion weight on VAL, then blend the TEST CSVs at it
#                                                  -> solution_blend_best3_eqm.csv
# Ensembles/blends are CSV-level by id (same validated approach as the 2.98
# blend). Per-job scratch so it can't collide with the still-running 10k job.
set -e
ROOT=$HOME/network_ml_project
SCRATCH="/scratch/izar/rgautam/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p $SCRATCH
rsync -a --delete --exclude='.venv' $ROOT $SCRATCH
echo "SYNCHRONIZED AT $(date) -> $SCRATCH"
trap "rm -rf $SCRATCH" EXIT

cd $SCRATCH/network_ml_project/src/mart
source $ROOT/.venv/bin/activate
SROOT="$SCRATCH/network_ml_project"
TEST="$SROOT/data/test/test"
SUB="$SROOT/submissions"
SNAP="$SROOT/src/mart/checkpoints/mart_aug_iso_hoops_sgdr_snapshots"

# (1) SGDR best single model
python submit_nba_pt.py \
    --checkpoint "$SROOT/src/mart/checkpoints/mart_aug_iso_hoops_sgdr_best.ckpt" \
    --test_dir "$TEST" --out_csv "$SUB/solution_sgdr_best_val3.116.csv"

# (2) best-3 member CSVs on test
for c in 07 08 09; do
  python submit_nba_pt.py \
      --checkpoint "$SNAP/cycle_${c}.ckpt" \
      --test_dir "$TEST" --out_csv "$SUB/_snap_cycle_${c}.csv"
done
# best-3 ensemble = equal-weight mean of the three cycle CSVs
python blend_csvs.py --out "$SUB/solution_sgdr_snap_best3_val3.101.csv" \
    --inputs "$SUB/_snap_cycle_07.csv" "$SUB/_snap_cycle_08.csv" "$SUB/_snap_cycle_09.csv"

# (3) tune best-3 x EqMotion weight on VAL, capture the chosen weight
echo "==================== TUNE best-3 x EqMotion (val) ===================="
TUNE_OUT=$(python tune_best3_eqm_blend.py \
    --snapshot_dir "$SNAP" --cycles 07 08 09 \
    --eqm_cache "$SROOT/cache/eqm_residual" \
    --split_path "$SROOT/splits/fold0.json")
echo "$TUNE_OUT"
W=$(echo "$TUNE_OUT" | grep -oP 'CHOSEN_WEIGHT=\K[0-9.]+')
echo "[INFO] chosen w_best3 = $W"

# Apply that weight to the TEST blend: W*best3 + (1-W)*EqMotion ensemble CSV.
WEQM=$(python -c "print(round(1.0 - $W, 4))")
python blend_csvs.py --out "$SUB/solution_blend_best3_eqm_w${W}.csv" \
    --inputs "$SUB/solution_sgdr_snap_best3_val3.101.csv" \
             "$SUB/solution_ens5_iso_hoops_val3.25.csv" \
    --weights "$W" "$WEQM"

# Copy the three FINAL submissions back to HOME (skip _snap_* intermediates).
mkdir -p $ROOT/submissions
rsync -a "$SUB/solution_sgdr_best_val3.116.csv" \
         "$SUB/solution_sgdr_snap_best3_val3.101.csv" \
         "$SUB/solution_blend_best3_eqm_w${W}.csv" \
         $ROOT/submissions/ && echo "Copied 3 submissions to HOME"
echo "--- HOME submissions ---"; ls -la $ROOT/submissions/solution_sgdr_*.csv $ROOT/submissions/solution_blend_best3_*.csv
echo "==================== DONE ($(date)) ===================="
