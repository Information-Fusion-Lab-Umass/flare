#!/bin/bash
#
#SBATCH --job-name=joies_job
#SBATCH -e profiler.err            # File to which STDERR will be written
#SBATCH --output=profiler.txt  #%j.txt output file
#SBATCH --partition=m40-long    # Partition to submit to
#
#SBATCH --ntasks=5
#SBATCH --time=02-01:00         # Runtime in D-HH:MM
#SBATCH --mem=50GB

python profile_example.py
sleep 1
exit


