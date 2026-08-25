#!/bin/bash -l
#SBATCH --job-name=test_resume_wandb
#SBATCH --time=00:25:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=ICT26_MHPC_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/slurm/%x/%j.err

# Regression test for checkpoint resume + W&B offline-run continuation (see
# get_or_create_wandb_run_id in walrus/train.py and the "Resuming vs. starting
# fresh" section of README.md).
#
# Runs the tiny data=debug/model=debug/trainer=debug config four times in a
# row under the same experiment name, back to back in this one job - one
# fresh run plus three resumes (MAX_EPOCHS below), each training one further
# epoch than the last:
#   phase 1: auto_resume=False, trainer.max_epoch=1  -> fresh run
#   phase 2: auto_resume=True,  trainer.max_epoch=2  -> resume to epoch 2
#   phase 3: auto_resume=True,  trainer.max_epoch=3  -> resume to epoch 3
#   phase 4: auto_resume=True,  trainer.max_epoch=4  -> resume to epoch 4
# A later process picking up a checkpoint left behind by an earlier one is
# exactly what happens when a real job times out and the script is
# resubmitted - this just triggers that path directly, three times in a row,
# instead of waiting out three real time limits. Uses a dedicated
# `logger.wandb_project_name` so test runs don't clutter real experiment
# tracking.
#
# Usage: ./submit.sh test_resume_wandb.sh
# Safe to rerun: phase 1 always forces a brand-new run_idx folder
# (auto_resume=False), so repeated runs never collide with each other.
# Leftover run folders under logs/test_resume_wandb-.../ and the phase logs
# under logs/test_resume_wandb/ are disposable - delete them whenever.
#
# The job itself can only prepare four offline W&B run directories sharing
# one run id - compute nodes have no outbound internet, so the actual sync
# (and the "is this really one continuous run" check) has to happen by hand
# from the login node afterward. The script prints the exact command plus
# what to look for once it finishes.
#
# Sharing a run id alone is not sufficient for the synced history to actually
# read as one continuous run: each phase is a fresh process, so without an
# explicit `step=`, wandb.log()'s internal step counter restarts at 0 every
# phase - all four phases would collide at step 0 and overwrite each other
# instead of appending. Trainer.train() now passes step=epoch to every
# wandb.log() call (see walrus/trainer/training.py) specifically so each
# phase writes to its own non-overlapping global step; the --append flag on
# the sync command below then tells the server to attach that history to the
# existing run instead of starting a new one.

set -uo pipefail
# Deliberately no -e: each phase's exit status is checked explicitly below so
# a failure can be reported with context instead of the script dying silently
# mid-pipe.

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs

module purge
module load python
module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate

cd /leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus

EXPERIMENT_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs
TEST_LOG_DIR=$EXPERIMENT_DIR/test_resume_wandb
mkdir -p "$TEST_LOG_DIR"

COMMON_ARGS=(distribution=ddp server=leonardo data=debug model=debug trainer=debug
    ++experiment_dir=$EXPERIMENT_DIR name=test_resume_wandb
    logger.wandb_project_name=walrus_leonardo_test data_workers=2)

# One fresh run (auto_resume=False) followed by three resumes
# (auto_resume=True), each trained one epoch further than the last.
MAX_EPOCHS=(1 2 3 4)
PHASE_LOGS=()

for i in "${!MAX_EPOCHS[@]}"; do
    phase_num=$((i + 1))
    max_epoch=${MAX_EPOCHS[$i]}
    if [[ $phase_num -eq 1 ]]; then
        auto_resume=False
        label="fresh run"
    else
        auto_resume=True
        label="resume (simulated resubmission $((phase_num - 1)))"
    fi
    phase_log="$TEST_LOG_DIR/phase${phase_num}_${SLURM_JOB_ID}.log"
    PHASE_LOGS+=("$phase_log")

    echo
    echo "=== Phase $phase_num: $label, train to epoch $max_epoch ==="
    srun python -u `which torchrun` \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=${SLURM_JOB_ID}_phase${phase_num} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$SLURMD_NODENAME:29500 \
            train.py "${COMMON_ARGS[@]}" auto_resume=$auto_resume trainer.max_epoch=$max_epoch \
        2>&1 | tee "$phase_log"
    phase_status=${PIPESTATUS[0]}
    if [[ $phase_status -ne 0 ]]; then
        echo "FAIL: phase $phase_num training run itself failed (exit $phase_status) - see $phase_log"
        exit 1
    fi
done

echo
echo "=== Verifying continuation ==="
STATUS=0
N_PHASES=${#MAX_EPOCHS[@]}

EXP_FOLDERS=()
for phase_log in "${PHASE_LOGS[@]}"; do
    EXP_FOLDERS+=("$(grep -m1 -oP '(?<=Using default experiment folder ).+' "$phase_log")")
done
EXP_FOLDER_1="${EXP_FOLDERS[0]}"

FOLDERS_CONSISTENT=1
for i in "${!EXP_FOLDERS[@]}"; do
    if [[ -z "${EXP_FOLDERS[$i]}" ]]; then
        echo "FAIL: could not determine experiment folder for phase $((i + 1))."
        STATUS=1
        FOLDERS_CONSISTENT=0
    elif [[ "${EXP_FOLDERS[$i]}" != "$EXP_FOLDER_1" ]]; then
        echo "FAIL: phase $((i + 1)) used a different experiment folder (${EXP_FOLDERS[$i]}) than phase 1 ($EXP_FOLDER_1)."
        STATUS=1
        FOLDERS_CONSISTENT=0
    fi
done
if [[ $FOLDERS_CONSISTENT -eq 1 ]]; then
    echo "PASS: all $N_PHASES phases used the same experiment folder: $EXP_FOLDER_1"
fi

if [[ -n "$EXP_FOLDER_1" && ( -L "$EXP_FOLDER_1/checkpoints/last" || -d "$EXP_FOLDER_1/checkpoints/last" ) ]]; then
    echo "PASS: checkpoints/last exists in the experiment folder."
else
    echo "FAIL: checkpoints/last not found under $EXP_FOLDER_1."
    STATUS=1
fi

for i in "${!MAX_EPOCHS[@]}"; do
    phase_num=$((i + 1))
    if [[ $phase_num -eq 1 ]]; then
        continue
    fi
    prev_epoch=${MAX_EPOCHS[$((i - 1))]}
    phase_log="${PHASE_LOGS[$i]}"
    if grep -q "Resume from checkpoint" "$phase_log" && grep -q "Resume from epoch $prev_epoch " "$phase_log"; then
        echo "PASS: phase $phase_num log shows checkpoint resume from epoch $prev_epoch."
    else
        echo "FAIL: phase $phase_num log does not show expected checkpoint-resume from epoch $prev_epoch."
        STATUS=1
    fi
done

WANDB_ID=""
if [[ -n "$EXP_FOLDER_1" && -f "$EXP_FOLDER_1/wandb_run_id.txt" ]]; then
    WANDB_ID=$(cat "$EXP_FOLDER_1/wandb_run_id.txt")
    N_OFFLINE_DIRS=$(ls -d "$WANDB_DIR"/wandb/offline-run-*-"$WANDB_ID" 2>/dev/null | wc -l)
    if [[ "$N_OFFLINE_DIRS" -eq "$N_PHASES" ]]; then
        echo "PASS: all $N_PHASES phases wrote a W&B offline run sharing id $WANDB_ID."
    else
        echo "FAIL: expected $N_PHASES offline-run dirs sharing id $WANDB_ID, found $N_OFFLINE_DIRS."
        STATUS=1
    fi
else
    echo "FAIL: wandb_run_id.txt not found under $EXP_FOLDER_1."
    STATUS=1
fi

echo
if [[ "$STATUS" -eq 0 ]]; then
    echo "ALL CHECKS PASSED."
else
    echo "SOME CHECKS FAILED - see above, and inspect the phase logs under $TEST_LOG_DIR."
fi

if [[ -n "$WANDB_ID" ]]; then
    echo
    echo "To confirm on wandb.ai too, from the login node:"
    echo "  cd $WANDB_DIR && wandb sync --sync-all --append --include-offline --include-globs=\"run-$WANDB_ID.wandb\""
    echo "  (--sync-all ignores any path argument and always scans ./wandb relative to the"
    echo "  current directory, hence the cd; --include-globs matches the .wandb filename"
    echo "  *inside* each offline-run-*/ dir, not the offline-run-*/ dir name itself - this"
    echo "  scopes the sync to just this test's $N_PHASES runs, rather than sweeping in every"
    echo "  other unsynced real run sitting in the same shared WANDB_DIR; --append tells the"
    echo "  server to attach each dir's history to the existing run id instead of creating a"
    echo "  fresh one - Trainer.train()'s wandb.log(..., step=epoch) calls already give every"
    echo "  phase its own non-overlapping global step, so this is what actually makes the"
    echo "  history line up into one continuous run rather than 4 runs colliding at step 0)."
    echo "  Then open project walrus_leonardo_test and check run id $WANDB_ID:"
    echo "  it should be a SINGLE run whose train/epoch metric rises continuously"
    echo "  from 1 to ${MAX_EPOCHS[$((N_PHASES - 1))]} (not $N_PHASES separate runs)."
fi
exit $STATUS
