#!/bin/bash

#SBATCH --job-name=mart_curriculum_5k
#SBATCH --output=/home/kolhe/network_ml_project/jobs/out/%x_%j.out
#SBATCH --error=/home/kolhe/network_ml_project/jobs/out/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=ee-452

SCRATCH="/scratch/izar/kolhe/job_$SLURM_JOB_ID"
rsync -a --exclude='__pycache__' --exclude='*.db' --exclude='logs/' \
      $HOME/network_ml_project $SCRATCH
echo "SYNCHRONIZED AT $(date)"

cd $SCRATCH/network_ml_project/src/mart

source /home/kolhe/miniconda3/etc/profile.d/conda.sh
conda activate nanovlm
pip install python-box python-dotenv wandb -q

python main_nba_pt.py \
    --config configs/mart_nba_aug.yaml \
    --split_path ../../splits/fold0.json \
    --model_name mart_curriculum_5k \
    --num_epochs 5000 \
    --iso_norm \
    --aug_rot_deg 180 \
    --aug_court_mirror \
    --use_hoops \
    --loss min_ade \
    --curriculum \
    --curriculum_switch 1500 \
    --gpu 0 \
    --wandb_run_name mart_curriculum_5k \
    --wandb_project NML_base

rsync -a checkpoints/ "$HOME/network_ml_project/src/mart/checkpoints/" 2>/dev/null \
    && echo "Checkpoints synced."
echo "DONE AT $(date)"
