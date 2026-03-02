#!/bin/bash --login
#
#SBATCH --job-name=dlrm_run_cores
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
CORES=${SLURM_CPUS_PER_TASK}
python tests/dlrm_comp_merged.py > results/dlrm_comp_merged_${CORES}.out
tail -n 8 results/dlrm_comp_merged_${CORES}.out > results/dlrm_perf_${CORES}.log