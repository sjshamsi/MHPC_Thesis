#!/bin/bash -l
#SBATCH --job-name=test_flops_fsdp
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/leonardo_work/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_work/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.err

# FLOP accounting on GPU under fsdp (1 node x 4 GPUs), the setup the CPU tests can't cover:
# hooks on FSDP-wrapped encoders, the all-reduce over 4 ranks, the recompute term, and the
# counters continuing across a resume. One run, two processes:
#   phase 1: auto_resume=False, max_epoch=2   epochs 1-2 (fresh run folder and W&B id)
#   phase 2: auto_resume=True,  max_epoch=4   epochs 3-4 (resumes phase 1)
# Then check_flops_fsdp.py compares the offline W&B history, every checkpoint's metadata.pt,
# and Tools/flop_planner/count_WALRUS_flops.py --exact (same overrides, overrides.txt).
#
# Usage (submit.sh creates the --output folder, which SLURM won't):
#   Tools/slurm/submit.sh Runs/test_flops_fsdp/test_flops_fsdp.sh
# Expect "ALL PASS" at the end of the .out file.

set -uo pipefail

export OMP_NUM_THREADS=8
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_work/ICT26_MHPC_0/sshamsi/logs  # runs land in logs/wandb/

module purge
module load python/3.11.7
module load cuda/12.6
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env2/bin/activate

REPO=/leonardo/home/userexternal/sshamsi0/Public_Thesis
TEST_DIR=$REPO/Runs/test_flops_fsdp
EXPERIMENT_DIR=/leonardo_work/ICT26_MHPC_0/sshamsi/logs
LOG_DIR=$EXPERIMENT_DIR/test_flops_fsdp
mkdir -p "$LOG_DIR"
mapfile -t OVERRIDES < <(grep -vE '^\s*(#|$)' "$TEST_DIR/overrides.txt")

cd "$REPO/walrus"
for phase in 1 2; do
    if [[ $phase -eq 1 ]]; then auto_resume=False; max_epoch=2; else auto_resume=True; max_epoch=4; fi
    echo "=== Phase $phase: auto_resume=$auto_resume, max_epoch=$max_epoch ==="
    srun python -u "$(which torchrun)" \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=${SLURM_JOB_ID}_$phase \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$SLURMD_NODENAME:29500 \
            train.py "${OVERRIDES[@]}" \
            ++experiment_dir=$EXPERIMENT_DIR \
            trainer.max_epoch=$max_epoch auto_resume=$auto_resume \
            hydra.run.dir=$LOG_DIR/hydra/${SLURM_JOB_ID}_phase$phase \
        2>&1 | tee "$LOG_DIR/phase${phase}_${SLURM_JOB_ID}.log"
    status=${PIPESTATUS[0]}
    if [[ $status -ne 0 ]]; then
        echo "FAIL: phase $phase training exited with $status (log: $LOG_DIR/phase${phase}_${SLURM_JOB_ID}.log)"
        exit 1
    fi
done

cd "$REPO"
python -W ignore "$TEST_DIR/check_flops_fsdp.py" "$LOG_DIR/phase1_${SLURM_JOB_ID}.log"
