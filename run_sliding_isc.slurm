#!/bin/bash

# Name of job?
#SBATCH --job-name=sliding_isc

# Where to output log files?
# make sure this logs directory exists!! otherwise the script won't run
#SBATCH --output='/jukebox/norman/rsiyer/isc/toystory/logs/%A.log'

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 00:01:00

# How much memory to allocate (in MB)?
##SBATCH --cpus-per-task=16 --mem-per-cpu=10000
#SBATCH --cpus-per-task=1 --mem-per-cpu=2000

# Update with your email 
##SBATCH --mail-user=rsiyer@princeton.edu
##SBATCH --mail-type=BEGIN,END,FAIL

conda init
conda activate isc2
cd /jukebox/norman/rsiyer/isc/isc_scripts

python sliding_isc_toystory.py
