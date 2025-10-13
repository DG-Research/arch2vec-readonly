#!/bin/bash

# The project ID which this job should run under:
#SBATCH --account="punim1875"
# The name of the job:
#SBATCH --job-name="generate-nb301"
# Partition for the job:
#SBATCH --partition feit-gpu-a100
#SBATCH --qos feit
# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=12:00:00
# Send yourself an email when the job:
#SBATCH --mail-user=dpoddenige@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
module load Anaconda3/2022.10
module load CUDA/12.2.0

export CONDA_ENVS_PATH=/data/gpfs/projects/punim1875/anaconda3/envs
export CONDA_PKGS_DIR=/data/gpfs/projects/punim1875/anaconda3/pkgs
eval "$(conda shell.bash hook)"
conda activate nas-vqvae-conda

python /data/gpfs/projects/punim1875/arch2vec-readonly/preprocessing/gen_isomorphism_graphs.py

