#!/bin/bash --login
#
#SBATCH --job-name=llm_run_cores
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
PWD_VAR=$(pwd)
cd workloads/InfiniGen-IBP; make -s run_small_expt OUTPUT=${PWD_VAR}/results/llm_output_${SLURM_CPUS_PER_TASK}.log > ${PWD_VAR}/results/llm_latency_${SLURM_CPUS_PER_TASK}.log;