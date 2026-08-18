#!/bin/bash
# Polls squeue on an interval and resubmits any of the given scripts once their
# named job has no PENDING/RUNNING instance left for this user - i.e. only once
# the previous attempt has genuinely finished (completed, crashed, or hit the
# time limit), never pre-queued ahead of time. This matters because
# boost_qos_dbg caps MaxSubmitPU at 2 total submitted jobs per user - if two
# scripts are running concurrently (already at the cap), queuing a replacement
# for either one *before* it exits would need a 3rd slot and get rejected with
# QOSMaxSubmitJobPerUserLimit, same failure seen earlier resubmitting
# halfDims_allDsets_tiny/_tiny_video by hand. Checking via squeue first avoids
# that race entirely.
#
# Each script's own AUTO_RESUME=True default (see the scripts themselves)
# handles the actual training continuation via the trainer's checkpoint - this
# watcher only automates *submitting the next SLURM job*, nothing about the
# training resume logic itself.
#
# Stops resubmitting a given script once its own training has already reached
# max_epoch and finished the mandatory final test pass, detected by grepping
# that script's most recent .out log for "rollout test loss" (the last line
# Trainer.train() ever logs - see walrus/trainer/training.py's
# validate_if_necessary, only reached once after the full epoch loop
# completes).
#
# Usage: run directly in an interactive terminal (a plain foreground process -
# no nohup/disown, so it only keeps running while this shell/session does; run
# it inside `screen`/`tmux` if you want to detach your terminal but still have
# it tied to that session):
#   cd LeonardoRunScripts
#   ./watch_and_resubmit.sh halfDims_activematter_BS1000.sh halfDims_allDsets_BS1000.sh
#
# Stop it any time with Ctrl-C.
#
# This is a lightweight polling loop (one squeue + a couple of small greps per
# script per interval) - it does no heavy computation itself, only launches
# sbatch jobs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR=/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs/slurm
POLL_INTERVAL_SECONDS=300

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <script1.sh> [script2.sh ...]" >&2
    exit 1
fi

SCRIPTS=("$@")

trap 'echo; echo "$(date "+%F %T") Stopping watcher (Ctrl-C) - any already-submitted jobs keep running."; exit 0' INT TERM

echo "$(date '+%F %T') Starting watcher for: ${SCRIPTS[*]} (polling every ${POLL_INTERVAL_SECONDS}s). Ctrl-C to stop."

while true; do
    for script in "${SCRIPTS[@]}"; do
        script_path="$SCRIPT_DIR/$script"
        if [[ ! -f "$script_path" ]]; then
            echo "$(date '+%F %T') WARNING: $script_path not found, skipping."
            continue
        fi

        job_name=$(grep -m1 -oP '(?<=#SBATCH --job-name=).+' "$script_path")
        if [[ -z "$job_name" ]]; then
            echo "$(date '+%F %T') WARNING: could not find --job-name in $script, skipping."
            continue
        fi

        # Still pending or running under this name - nothing to do yet.
        if squeue -u "$USER" -n "$job_name" -h 2>/dev/null | grep -q .; then
            continue
        fi

        # Already fully finished (reached max_epoch + final test pass)?
        latest_out=$(ls -t "$LOGS_DIR/$job_name"/*.out 2>/dev/null | head -n1)
        if [[ -n "$latest_out" ]] && grep -q "rollout test loss" "$latest_out"; then
            echo "$(date '+%F %T') $job_name: training already complete (found in $latest_out), not resubmitting."
            continue
        fi

        echo "$(date '+%F %T') $job_name: no active job and not yet complete, resubmitting."
        (cd "$SCRIPT_DIR" && ./submit.sh "$script")
    done
    sleep "$POLL_INTERVAL_SECONDS"
done
