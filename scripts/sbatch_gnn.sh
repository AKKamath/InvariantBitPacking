#!/bin/bash --login
#
#SBATCH --job-name=gnn_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --mem=300GB
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=mcnode22

#set -e # stop bash script on first error

conda activate ibp
make gnn