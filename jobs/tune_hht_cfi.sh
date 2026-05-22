#!/bin/bash

#SBATCH --job-name=hht_tune
#SBATCH --output=/home/kolhe/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/kolhe/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

SCRATCH="/scratch/izar/kolhe/"
rsync -a --delete --exclude='__pycache__' --exclude='*.db' --exclude='logs/' \
      $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project

source /home/kolhe/miniconda3/etc/profile.d/conda.sh
conda activate nanovlm

# Copy optuna DB from home if it exists (allows resuming a previous study)
if [ -f "$HOME/network_ml_project/optuna_hht_cfi.db" ]; then
    cp "$HOME/network_ml_project/optuna_hht_cfi.db" .
    echo "Resumed existing Optuna study from home."
fi

python src/hht_cfi/tune_hht_cfi.py

# Copy results back: DB + logs + best checkpoint (if any)
cp optuna_hht_cfi.db "$HOME/network_ml_project/" 2>/dev/null && echo "DB copied back."
rsync -a logs/ "$HOME/network_ml_project/logs/" 2>/dev/null

echo "DONE AT $(date)"
python src/hht_cfi/tune_hht_cfi.py --report
