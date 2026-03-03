#!/bin/bash 

#SBATCH --job-name=preprocessTest
#SBATCH --acount=462001312
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=small

module load cray-python

uv sync

srun .venv/bin/python test.py



