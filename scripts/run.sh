#!/bin/bash
#
#SBATCH --job-name=featcat_wt3_exp4_20000
#SBATCH -e errors/res_%j.err            # File to which STDERR will be written
#SBATCH --output=run_output/res_%j.txt     # output file
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --ntasks=5
#SBATCH --time=02-01:00         # Runtime in D-HH:MM
#SBATCH --mem=30GB

python main.py --config=config/config_tadpole_clf2_featcat.yaml
sleep 1
exit


