#!/bin/bash

# The project ID which this job should run under:
#SBATCH --account="punim1875"
# The name of the job:
#SBATCH --job-name="pretrain-nb201"
# Partition for the job:
#SBATCH --partition gpu-a100-short
#SBATCH --qos normal
# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-04:00:00
# Send yourself an email when the job:
#SBATCH --mail-user=dpoddenige@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

# The modules to load:
source ../envs/nas_venv/bin/activate
module load foss/2022a
module load PyTorch/1.12.1-CUDA-11.7.0
module load NCCL/2.12.12-CUDA-11.7.0
module load matplotlib/3.5.2

export NCCL_SOCKET_IFNAME="bond0.3027,p1p1.3027"
export NCCL_IB_HCA=mlx5_bond_0,mlx5_0
export NCCL_IB_GID_INDEX=7


python run_scripts/summarize_nb201_runtime.py \
    --raw-dir /data/gpfs/projects/punim1875/arch2vec-readonly/results/runtime_report_nb201_arch2vec/cuda/raw/ \
    --out-dir /data/gpfs/projects/punim1875/arch2vec-readonly/results/runtime_report_nb201_arch2vec/cuda/summary/