#!/bin/bash -l
#SBATCH --job-name=walrus_storage_bench_train
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# The authoritative comparison: runs the actual walrus train.py + MixedWellDataModule
# DataLoader (not a synthetic proxy) against a small active_matter subset staged
# under each of the 4 candidate `well_base_path` locations in turn, on the same
# GPU/node, back to back, so storage location is the only thing that changes.
# Uses model=debug (hidden_dim=8) so GPU compute time is negligible - any gap in the
# per-batch "data <x>s" timing that walrus/trainer/training.py logs (Trainer.run_epoch,
# also aggregated into avg_data_loading_time) is attributable to storage, not the model.
#
# NOTE (2026-08-24): `cinQuota` shows /leonardo_scratch/fast/ICT26_MHPC_0 already
# OVER its 1T quota (grace period). The staged subset here is ~1.5GB and is deleted
# at the end of each path's block, but if the quota issue isn't resolved by the time
# you run this, drop that path from PATHS below.
#
# After the job finishes, compare average data-loading time per path with e.g.:
#   grep -A0 'Data:' logs/slurm/walrus_storage_bench_train/<jobid>.out \
#       | grep -oP 'data \K[0-9.]+(?=s,)' | awk '{s+=$1; n++} END {print s/n}'
# run once per path's section of the log (the "STORAGE_BENCH_WALRUS target:" banners
# below mark where each path's section starts).

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus

SRC=/leonardo_work/ICT26_MHPC_0/sshamsi/datasets/active_matter
TRAIN_FILE=$(ls "$SRC"/data/train/*.hdf5 | head -1)
VALID_FILE=$(ls "$SRC"/data/valid/*.hdf5 | head -1)
TEST_FILE=$(ls "$SRC"/data/test/*.hdf5 | head -1)

PATHS=(
    /leonardo_work/ICT26_MHPC_0/sshamsi/storage_bench_walrus
    /leonardo_work/ICT26_MHPC/sshamsi/storage_bench_walrus
    /leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/storage_bench_walrus
    /leonardo_scratch/fast/ICT26_MHPC/sshamsi/storage_bench_walrus
)

for i in "${!PATHS[@]}"; do
    p="${PATHS[$i]}"
    port=$((29500 + i))

    echo "=========================================================="
    echo "STORAGE_BENCH_WALRUS target: $p"
    echo "=========================================================="

    mkdir -p "$p/active_matter/data/train" "$p/active_matter/data/valid" "$p/active_matter/data/test"
    cp -n "$SRC/stats.yaml" "$p/active_matter/"
    cp -n "$TRAIN_FILE" "$p/active_matter/data/train/"
    cp -n "$VALID_FILE" "$p/active_matter/data/valid/"
    cp -n "$TEST_FILE" "$p/active_matter/data/test/"

    srun python -u `which torchrun` \
        --nnodes=1 \
        --nproc_per_node=1 \
        --rdzv_id=${SLURM_JOB_ID}_${i} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$SLURMD_NODENAME:$port \
            train.py distribution=ddp server=leonardo data=debug model=debug trainer=debug \
                ++data.well_base_path="$p/" \
                ++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs \
                name=storage_bench_walrus_path${i} \
                logger.wandb_project_name=walrus_leonardo_test \
                data_workers=8 data.module_parameters.batch_size=8 \
                data.module_parameters.max_samples=200 \
                trainer.max_epoch=3 trainer.log_interval=1 \
                auto_resume=False

    rm -rf "$p"
done
