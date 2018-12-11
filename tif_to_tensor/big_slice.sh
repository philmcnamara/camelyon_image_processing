#! /bin/bash
#SBATCH --partition=short
#SBATCH --job-name=big_slice
#SBATCH --output=slice.out
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=phil.j.mcnamara@gmail.com

ml anaconda3

mkdir /tmp/sliced

./image_slice.py

# Move everything from temp to our shared folder
cp /tmp/sliced/* /projects/bgmp/oda/training/sliced/tumor_001/

rm -r /tmp/sliced
