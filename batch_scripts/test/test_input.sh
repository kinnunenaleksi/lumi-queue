#!/bin/bash 

#SBATCH --job-name=preprocessTest
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --error=runs/R-%x.%j.err
#SBATCH --account=project_462001312
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=debug

module load cray-python

source .venv/bin/activate

$HOME/.local/bin/uv sync

$HOME/.local/bin/uv pip install -e .

srun python batch_scripts/test_input.py
