#!/bin/bash

#SBATCH --job-name=trainRegressStandard
#SBATCH --output=runs/r-%x.%j.out
#SBATCH --error=runs/r-%x.%j.err
#SBATCH --account=project_462001312
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=standard

module load cray-python

source .venv/bin/activate

$HOME/.local/bin/uv sync

$HOME/.local/bin/uv pip install -e .

srun python batch_scripts/scripts/training/regress/train_regress_filtered.py
