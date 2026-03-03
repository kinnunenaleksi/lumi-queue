#!/bin/bash 

#SBATCH --job-name=mlTest
#SBATCH --output=runs/R-%x.%j.out
#SBATCH --error=runs/R-%x.%j.err
#SBATCH --account=project_462001312
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=small

module load cray-python

source .venv/bin/activate

$HOME/.local/bin/uv sync

srun python batch_scripts/python test_ml.py

