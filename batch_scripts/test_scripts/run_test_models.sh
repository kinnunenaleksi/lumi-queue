#!/bin/bash

#SBATCH --job-name=testRunModels
#SBATCH --output=runs/r-%x.%j.out
#SBATCH --error=runs/r-%x.%j.err
#SBATCH --account=project_462001312
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=small-g

module load cray-python

source .venv/bin/activate

$HOME/.local/bin/uv sync

$HOME/.local/bin/uv pip install -e .

srun python batch_scripts/scripts/test_scripts/run_test_models.py
