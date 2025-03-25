#!/bin/bash

# Name of job?
#SBATCH --job-name=sliding_isc

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='/jukebox/norman/rsiyer/isc/toystory/logs/%A.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 48:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=24000

# Update with your email 
##SBATCH --mail-user=rsiyer@princeton.edu
##SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load anacondapy/2024.02
conda activate /usr/people/ri4541/miniconda3/envs/isc2
cd /jukebox/norman/rsiyer/isc/isc_scripts
python sliding_isc_toystory.py
jupyter nbconvert --to script sliding_isc.ipynb --output ridge && ipython ridge.py
