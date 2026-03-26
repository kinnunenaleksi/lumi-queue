#!/bin/bash

#SBATCH --job-name=testAblationModels
#SBATCH --output=runs/r-%x.%j.out
#SBATCH --error=runs/r-%x.%j.err
#SBATCH --account=project_462001312
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=64

module load cray-python

source .venv/bin/activate

$HOME/.local/bin/uv sync

$HOME/.local/bin/uv pip install -e .

srun python batch_scripts/scripts/test_ablation.py
