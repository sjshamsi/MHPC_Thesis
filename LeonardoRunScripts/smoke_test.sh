#!/bin/bash -l
#SBATCH --job-name=walrus_smoke_test
#SBATCH --time=00:20:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x-%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm/%x-%j.err

# Fast end-to-end pipeline check: tiny model + tiny active_matter subset (data=debug,
# model=debug), 1 epoch. Confirms modules/venv/torchrun/data-path/checkpointing/logging
# all work before spending real GPU-hours on LeonardoRunScripts/gen_scaling_runs.py jobs.

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/Walrus/walrus

srun python -u `which torchrun` \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
        train.py distribution=ddp server=leonardo data=debug model=debug trainer=debug \
            ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs \
            name=smoke_test logger.wandb_project_name=walrus_leonardo \
            data_workers=2 auto_resume=False
