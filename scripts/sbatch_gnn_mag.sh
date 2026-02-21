#!/bin/bash --login
#
#SBATCH --job-name=gnn_run_mag
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --mem=300GB
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=mcnode22

#set -e # stop bash script on first error

eval "$(conda shell.bash hook)"
conda activate ibp
make gnn_mag