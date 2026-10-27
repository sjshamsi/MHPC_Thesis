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
#SBATCH --output=/leonardo_work/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.out
#SBATCH --error=/leonardo_work/ICT26_MHPC_0/sshamsi/logs/slurm/%x/%j.err

# Regression test for checkpoint auto-resume, W&B offline-run continuation and the
# final-epoch clamp (see get_or_create_wandb_run_id in walrus/train.py, and
# wandb.log(..., step=epoch) plus `epoch = min(epoch, self.max_epoch)` in
# Trainer.train() in walrus/trainer/training.py).
#
# Runs a tiny config five times in a row under the same experiment name, back to
# back in this one job:
#   phase 1: auto_resume=False, trainer.max_epoch=1  -> fresh run
#   phase 2: auto_resume=True,  trainer.max_epoch=2  -> resume to epoch 2
#   phase 3: auto_resume=True,  trainer.max_epoch=3  -> resume to epoch 3
#   phase 4: auto_resume=True,  trainer.max_epoch=4  -> resume to epoch 4
#   phase 5: auto_resume=True,  trainer.max_epoch=4  -> already finished: resumes at
#            epoch 5 > max_epoch, so the training loop never runs and only the final
#            test validation does (this is what the clamp is for. Without it the
#            test is logged at epoch 5, one past the last trained epoch).
# Each phase is a fresh process picking up the checkpoint the previous one left
# behind, which is what happens when a real job times out and is resubmitted.
#
# For speed: data=debug (active_matter, max_samples=8 micro-batches per epoch),
# model=debug (hidden_dim=8, 1 processor block), trainer=debug (no image/video
# validation), plus trainer.debug_mode=True, which caps every validation, test and
# rollout pass at short_validation_length batches instead of the full sets.
#
# Usage (submit.sh creates the --output folder, which SLURM won't):
#   Tools/slurm/submit.sh Runs/test_resume_wandb/test_resume_wandb.sh
#
# Should check online on wandb.ai afterwards

set -uo pipefail
# No -e: each phase's exit status is checked explicitly so failures are reported.

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_work/ICT26_MHPC_0/sshamsi/logs

module purge
module load python/3.11.7
module load cuda/12.6
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env2/bin/activate

cd /leonardo/home/userexternal/sshamsi0/Public_Thesis/walrus

EXPERIMENT_DIR=/leonardo_work/ICT26_MHPC_0/sshamsi/logs
TEST_LOG_DIR=$EXPERIMENT_DIR/test_resume_wandb
mkdir -p "$TEST_LOG_DIR"

COMMON_ARGS=(distribution=ddp server=leonardo data=debug model=debug trainer=debug
    ++experiment_dir=$EXPERIMENT_DIR name=test_resume_wandb
    logger.wandb_project_name=walrus_leonardo_test data_workers=2
    ++trainer.debug_mode=True
    trainer.short_validation_length=2
    trainer.max_rollout_steps=4
    ++trainer.skip_spectral_metrics=True)

# One fresh run, three resumes, then a resubmission of the finished run (clamp test).
MAX_EPOCHS=(1 2 3 4 4)
PHASE_LOGS=()

for i in "${!MAX_EPOCHS[@]}"; do
    phase_num=$((i + 1))
    max_epoch=${MAX_EPOCHS[$i]}
    if [[ $phase_num -eq 1 ]]; then
        auto_resume=False
        label="fresh run"
    elif [[ $max_epoch -eq ${MAX_EPOCHS[$((i - 1))]} ]]; then
        auto_resume=True
        label="resubmit of a finished run (clamp test)"
    else
        auto_resume=True
        label="resume (simulated resubmission $((phase_num - 1)))"
    fi
    phase_log="$TEST_LOG_DIR/phase${phase_num}_${SLURM_JOB_ID}.log"
    PHASE_LOGS+=("$phase_log")

    echo
    echo "=== Phase $phase_num: $label, max_epoch=$max_epoch ==="
    srun python -u `which torchrun` \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=${SLURM_JOB_ID}_phase${phase_num} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$SLURMD_NODENAME:29500 \
            train.py "${COMMON_ARGS[@]}" auto_resume=$auto_resume trainer.max_epoch=$max_epoch \
            hydra.run.dir=$TEST_LOG_DIR/hydra/${SLURM_JOB_ID}_phase${phase_num} \
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
LAST_PHASE_LOG="${PHASE_LOGS[$((N_PHASES - 1))]}"
FINAL_EPOCH=${MAX_EPOCHS[$((N_PHASES - 1))]}

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

# Clamp test: the last phase resubmits a finished run. It must not train, and its
# final test validation must be logged at the last trained epoch, not one past it.
if grep -q "starting training" "$LAST_PHASE_LOG"; then
    echo "FAIL: phase $N_PHASES trained, but the run had already reached max_epoch=$FINAL_EPOCH."
    STATUS=1
else
    echo "PASS: phase $N_PHASES did not train (run already finished)."
fi
if grep -q "Epoch $FINAL_EPOCH/$FINAL_EPOCH: starting test validation" "$LAST_PHASE_LOG"; then
    echo "PASS: phase $N_PHASES ran the test validation at epoch $FINAL_EPOCH (clamp works)."
elif grep -q "Epoch $((FINAL_EPOCH + 1))/$FINAL_EPOCH: starting test validation" "$LAST_PHASE_LOG"; then
    echo "FAIL: phase $N_PHASES ran the test validation at epoch $((FINAL_EPOCH + 1)), one past max_epoch (clamp missing)."
    STATUS=1
else
    echo "FAIL: no 'starting test validation' line found in $LAST_PHASE_LOG."
    STATUS=1
fi

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
    echo "To confirm on wandb.ai too, from the login node (needs wandb < 0.30 for --append):"
    echo "  cd $WANDB_DIR && wandb sync --sync-all --append --include-offline --include-globs=\"run-$WANDB_ID.wandb\""
    echo "  (--sync-all scans ./wandb relative to the current directory, hence the cd;"
    echo "  --include-globs limits the sync to this test's runs; --append attaches each"
    echo "  phase's history to the existing run id instead of creating a new run.)"
    echo "  Then open project walrus_leonardo_test and check run id $WANDB_ID:"
    echo "  it should be a SINGLE run whose epoch metric rises from 1 to $FINAL_EPOCH"
    echo "  without restarting (not $N_PHASES separate runs). Phase $N_PHASES adds no new"
    echo "  epoch: its test results go to step $FINAL_EPOCH, which W&B may drop as a repeat."
fi
exit $STATUS
