#!/bin/bash --login
#
#SBATCH --job-name=llm_run
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128B
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=mcnode22

#set -e # stop bash script on first error
eval "$(conda shell.bash hook)"
conda activate ibp
cd workloads/InfiniGen-IBP; make -s run_expt > ../../results/llm_latency.log;