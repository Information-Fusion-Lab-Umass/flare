#!/bin/bash
#
#SBATCH --job-name=T1c_fl
#SBATCH -o misc/run_outputs/%j.txt            # output file
#SBATCH -e misc/errors/%j.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=120000    # Memory in MB per cpu allocated
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00          # HH:MM:SS

python3 main.py --config=../configs/flare_skorch_covtest.yaml --debug=0 --numT=1 --n_iter=40 --exp_id=T2_confirm_split_dp_3
sleep 
exit
