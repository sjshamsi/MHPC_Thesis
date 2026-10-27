#!/bin/bash
# Lists, and with --delete removes, everything one run script left in the logs folder:
#   logs/slurm/<job-name>/            SLURM .out/.err
#   logs/<job-name>/                  phase logs of the Runs/test_* scripts
#   logs/<name>-*/                    experiment folders (checkpoints, viz, configs), where
#                                     <name> is the Hydra name= in the script or in an
#                                     overrides.txt next to it
#   logs/wandb/offline-run-*-<id>/    offline W&B runs: ids from the experiment folders'
#                                     wandb_run_id.txt, plus the "Run data is saved locally"
#                                     lines in the SLURM logs (validation runs write no id file)
# "<name>-*" would also catch another run named "<name>-something": read the list first.
#
# Usage:
#   Tools/slurm/cleanup_run.sh <run_script.sh>                 # list only
#   Tools/slurm/cleanup_run.sh <run_script.sh> --delete        # delete after typing 'yes'
#   Tools/slurm/cleanup_run.sh <run_script.sh> --delete --yes  # delete, no prompt

set -euo pipefail
shopt -s nullglob

LOGS_DIR=/leonardo_work/ICT26_MHPC_0/sshamsi/logs
usage="usage: $0 <run_script.sh> [--delete] [--yes]"

[[ $# -ge 1 ]] || { echo "$usage" >&2; exit 1; }
SCRIPT="$1"
shift
[[ -f "$SCRIPT" ]] || { echo "No such file: $SCRIPT" >&2; exit 1; }
DELETE=false
YES=false
for arg in "$@"; do
    case "$arg" in
        --delete) DELETE=true ;;
        --yes) YES=true ;;
        *) echo "$usage" >&2; exit 1 ;;
    esac
done

JOB_NAME=$(grep -m1 -oP '(?<=#SBATCH --job-name=)\S+' "$SCRIPT" || true)
case "$JOB_NAME" in
    "" | slurm | wandb | . | .. | */*)
        echo "Unusable '#SBATCH --job-name=' ('$JOB_NAME') in $SCRIPT" >&2; exit 1 ;;
esac

# Bare Hydra name= (not wandb_project_name= etc.), skipping comments, where
# "#SBATCH --job-name=" would also match.
sources=("$SCRIPT")
overrides="$(dirname "$SCRIPT")/overrides.txt"
[[ -f "$overrides" ]] && sources+=("$overrides")
HYDRA_NAME=$(grep -hv '^[[:space:]]*#' "${sources[@]}" | grep -m1 -oP '(?<![\w.])name=\S+' | cut -d= -f2- || true)

echo "Parsed from $SCRIPT: job name '$JOB_NAME', Hydra name '${HYDRA_NAME:-(none)}'"
if squeue -u "$USER" -n "$JOB_NAME" -h 2>/dev/null | grep -q .; then
    echo "WARNING: a job named '$JOB_NAME' is queued or running; scancel it first for a clean slate." >&2
fi

dirs=()
for d in "$LOGS_DIR/slurm/$JOB_NAME" "$LOGS_DIR/$JOB_NAME"; do
    [[ -d "$d" ]] && dirs+=("$d")
done
experiment_dirs=()
[[ -n "$HYDRA_NAME" ]] && experiment_dirs=("$LOGS_DIR/$HYDRA_NAME-"*/)
dirs+=("${experiment_dirs[@]}")

run_ids=()
for exp_dir in "${experiment_dirs[@]}"; do
    while IFS= read -r f; do run_ids+=("$(cat "$f")"); done \
        < <(find "$exp_dir" -maxdepth 2 -name wandb_run_id.txt)
done
for d in "$LOGS_DIR/slurm/$JOB_NAME" "$LOGS_DIR/$JOB_NAME"; do
    [[ -d "$d" ]] || continue
    while IFS= read -r w; do run_ids+=("${w##*-}"); done \
        < <(grep -rhoP '(?<=Run data is saved locally in )\S*/wandb/offline-run-\S+' "$d" || true)
done
for id in $(printf '%s\n' "${run_ids[@]}" | sort -u); do  # unquoted: drops empty ids
    dirs+=("$LOGS_DIR"/wandb/offline-run-*-"$id")
done

if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "Nothing found."
    exit 0
fi
printf '  %s\n' "${dirs[@]}"
if ! $DELETE; then
    echo "List only; rerun with --delete to remove these."
    exit 0
fi
if ! $YES; then
    read -r -p "Delete all of the above? Type 'yes': " answer
    [[ "$answer" == yes ]] || { echo "Aborted, nothing deleted."; exit 1; }
fi
rm -rf -- "${dirs[@]}"
echo "Deleted ${#dirs[@]} folders."
