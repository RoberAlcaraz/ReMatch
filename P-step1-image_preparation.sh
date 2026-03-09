#!/bin/bash
#SBATCH --job-name=train-imgprep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "PWD=$(pwd)"
echo "HOST=$(hostname)"

# ── Activate environment ────────────────────────────────────────
source /data/shared/miniforge3/etc/profile.d/conda.sh
conda activate /data/EEG/envs/rematch

which python
python -V

nvidia-smi || true
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# ── Run ─────────────────────────────────────────────────────────
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python scripts/P1-image_preparation.py

echo "Training Step 1 done. Images segmented and patterns extracted."
