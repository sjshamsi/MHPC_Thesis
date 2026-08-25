#!/bin/bash -l
#SBATCH --job-name=walrus_storage_bench
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# Raw + Well-HDF5-realistic I/O throughput across the 4 candidate dataset storage
# locations, run from a compute node (not the login node, so results reflect what
# an actual training job sees). Only requests 1 GPU/8 CPUs (not --exclusive) since
# no GPU compute happens here - this is a storage test, kept cheap on GPU-hours.
# See storage_bench_io.py's docstring for what each phase measures, and
# storage_bench_walrus.sh for the complementary real-training-run comparison.


set -euo pipefail

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

SCRIPT_DIR=/leonardo/home/userexternal/sshamsi0/MHPC_Thesis/LeonardoRunScripts

# One real, already-downloaded Well HDF5 file, staged into each candidate path in
# turn for the random-window-read phase.
SOURCE_FILE=/leonardo_work/ICT26_MHPC_0/sshamsi/datasets/active_matter/data/test/active_matter_L_10.0_zeta_11.0_alpha_-2.0.hdf5

PATHS=(
    /leonardo_work/ICT26_MHPC_0/sshamsi/storage_bench
    /leonardo_work/ICT26_MHPC/sshamsi/storage_bench
    /leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/storage_bench
    /leonardo_scratch/fast/ICT26_MHPC/sshamsi/storage_bench
)

for p in "${PATHS[@]}"; do
    python "$SCRIPT_DIR/storage_bench_io.py" \
        --path "$p" --source "$SOURCE_FILE" \
        --size-gb 4 --workers "${SLURM_CPUS_ON_NODE:-8}" --reads-per-worker 200
done
