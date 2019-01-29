#! /bin/bash

#SBATCH --partition=short
#SBATCH --job-name=multiProcessing
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

ml anaconda3

python  multi.py
