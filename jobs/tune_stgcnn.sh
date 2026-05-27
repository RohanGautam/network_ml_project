#!/bin/bash

#SBATCH --job-name=tune
#SBATCH --output=/home/rgautam/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/rgautam/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=team-ai

# NOTE: keep WALL_HOURS below in sync with --time above (SBATCH directives can't
# read shell variables). The tuning phase is capped at WALL_HOURS minus a fixed
# reserve so the top-K retrain + submission always finish before the wall.
WALL_HOURS=24
RETRAIN_RESERVE=$(( 2*3600 ))   # ~1h to retrain top-5 @250ep + TTA predict, with buffer


SCRATCH="/scratch/izar/rgautam/"
rsync -a --delete --exclude='.venv' $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

# Persistent Optuna study lives in $HOME so it survives /scratch wipes and resumes
# across re-launches of this job.
STUDY_DIR="$HOME/network_ml_project/optuna_studies"
mkdir -p "$STUDY_DIR"

cd $SCRATCH/network_ml_project
source $HOME/network_ml_project/.venv/bin/activate

# Cap the tuning phase so the retrain phase always has its reserve.
TUNE_TIMEOUT=$(( WALL_HOURS*3600 - RETRAIN_RESERVE ))

python src/stgcnn/tune_stgcnn.py \
    --study-name stgcnn_v1 \
    --study-path "$STUDY_DIR/stgcnn.db" \
    --n-trials 250 \
    --max-epochs 50 \
    --patience 10 \
    --timeout "$TUNE_TIMEOUT" \
    --top-k 5 \
    --retrain-epochs 250

# Preserve outputs in $HOME (scratch is wiped by the next job's rsync).
cp -f $SCRATCH/network_ml_project/submissions/solution_stgcnn_*.csv \
      $HOME/network_ml_project/submissions/ 2>/dev/null && echo "Copied submissions to HOME"
