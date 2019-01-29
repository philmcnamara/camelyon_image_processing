#! /bin/bash

#SBATCH --partition=long
#SBATCH --job-name=multiProcessing
#SBATCH --time=1-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

ml anaconda3

python oversample.py
