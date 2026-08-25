#!/bin/bash
# Deletes the SLURM log directory, experiment folder(s) (checkpoints, viz,
# profiler_traces, configs), and offline WandB run directories associated with a given
# LeonardoRunScripts/*.sh sbatch script - parses the script itself to find them, so it
# works for any run script in this directory, not just one hardcoded run.
# Safe by default: with no --delete it only lists what it *would* delete.
#
# Usage:
#   ./cleanup_run.sh <path/to/run_script.sh>                # dry run - lists paths, deletes nothing
#   ./cleanup_run.sh <path/to/run_script.sh> --delete        # deletes, after a y/n prompt
#   ./cleanup_run.sh <path/to/run_script.sh> --delete --yes  # deletes, no prompt
#
# How the paths are derived, and the one real collision risk:
#   - SLURM log dir: exact match on the script's "#SBATCH --job-name=" (see submit.sh,
#     which uses this same value for logs/slurm/<job_name>/). Exact directory name, not
#     a glob - no collision risk.
#   - Experiment folder(s): matched as "<hydra name=>-*", where "<hydra name=>" is the
#     script's bare Hydra "name=" override (the value passed to train.py, which becomes
#     cfg.name - NOT necessarily the same string as --job-name=, even though every
#     script in this repo currently keeps them equal). See
#     walrus/utils/experiment_utils.py::get_experiment_name: the actual folder name is
#     f"{cfg.name}-{data_name}-{prediction_type}-{model_name}[...]-{optimizer_name}-{lr}"
#     - a literal dash immediately after cfg.name, unconditionally, whenever
#     automatic_setup=True (every script's default). The trailing dash in the glob is
#     deliberate and load-bearing: several scripts in this repo share a literal name
#     prefix (fullDims_6DS / fullDims_6DS_N4 / fullDims_6DS_N8; halfDims_activematter /
#     _BS500 / _BS1000 / _multinode / _tiny_video), and "<name>*" (no dash required)
#     would also match those siblings' experiment folders and delete unrelated runs.
#     "<name>-*" does not, because none of those names contain a literal "-" of their
#     own - verified against every script and every experiment folder that exists in
#     this repo as of writing (zero cross-matches with the dash required, several
#     without it).
#     RESIDUAL RISK: if a future run script's name= contains a literal "-", or one
#     script's name is itself another script's name + "-" + suffix (e.g. "run" and
#     "run-large"), this boundary stops being sufficient to tell them apart. That's
#     exactly what the dry-run listing below is for - read it before passing --delete.
#   - WandB offline-run dir(s): found via each matched experiment folder's
#     wandb_run_id.txt (see walrus/train.py::get_or_create_wandb_run_id), matched by the
#     *entire* run id as the glob suffix - wandb run ids are generated to be unique, so
#     there's no meaningful collision risk here.

set -euo pipefail
shopt -s nullglob

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path/to/run_script.sh> [--delete] [--yes]" >&2
    exit 1
fi

SCRIPT_PATH="$1"
shift
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "No such file: $SCRIPT_PATH" >&2
    exit 1
fi

DELETE=false
ASSUME_YES=false
for arg in "$@"; do
    case "$arg" in
        --delete) DELETE=true ;;
        --yes) ASSUME_YES=true ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 <path/to/run_script.sh> [--delete] [--yes]" >&2
            exit 1
            ;;
    esac
done

LOGS_DIR="/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/logs"

SBATCH_JOB_NAME="$(grep -m1 -oP '(?<=#SBATCH --job-name=).+' "$SCRIPT_PATH" || true)"
if [ -z "$SBATCH_JOB_NAME" ]; then
    echo "Could not find '#SBATCH --job-name=' in $SCRIPT_PATH" >&2
    exit 1
fi

# Bare "name=<value>" Hydra override - the negative lookbehind excludes
# "wandb_project_name=", "data.wandb_data_name=", etc. (anything preceded by a word
# character or '.'), matching only a standalone "name=" token.
HYDRA_NAME="$(grep -oP '(?<![\w.])name=\S+' "$SCRIPT_PATH" | head -1 | sed 's/^name=//' || true)"
if [ -z "$HYDRA_NAME" ]; then
    echo "WARNING: no bare 'name=' Hydra override found in $SCRIPT_PATH - falling back" >&2
    echo "to the SBATCH job name ('$SBATCH_JOB_NAME') for the experiment folder too." >&2
    echo "Every script in this repo currently keeps these equal, but double-check the" >&2
    echo "listing below before deleting anything." >&2
    HYDRA_NAME="$SBATCH_JOB_NAME"
fi

echo "Parsed from $SCRIPT_PATH:"
echo "  SBATCH job name : $SBATCH_JOB_NAME"
echo "  Hydra name=     : $HYDRA_NAME"
echo

# Warn (don't block) if a job with this name is currently queued/running - deleting
# logs/checkpoints out from under a live job can confuse it (checkpointer writing into a
# directory that just vanished) or just leave you unable to inspect a run still in flight.
if squeue -u "$USER" -n "$SBATCH_JOB_NAME" -h 2>/dev/null | grep -q .; then
    echo "WARNING: a SLURM job named '$SBATCH_JOB_NAME' is currently queued/running:" >&2
    squeue -u "$USER" -n "$SBATCH_JOB_NAME" >&2
    echo "Consider 'scancel' first if you want a clean slate." >&2
    echo >&2
fi

slurm_dir="$LOGS_DIR/slurm/$SBATCH_JOB_NAME"
experiment_dirs=("$LOGS_DIR/${HYDRA_NAME}-"*/)

wandb_dirs=()
for exp_dir in "${experiment_dirs[@]}"; do
    while IFS= read -r id_file; do
        run_id="$(cat "$id_file")"
        [ -n "$run_id" ] || continue
        for wandb_dir in "$LOGS_DIR"/wandb/offline-run-*-"$run_id"; do
            wandb_dirs+=("$wandb_dir")
        done
    done < <(find "$exp_dir" -maxdepth 2 -name wandb_run_id.txt 2>/dev/null)
done

echo "=== Paths that will be targeted ==="
echo "--- SLURM log dir ---"
if [ -d "$slurm_dir" ]; then echo "$slurm_dir"; else echo "(none found)"; fi
echo "--- Experiment folder(s) (checkpoints, viz, profiler_traces, configs) ---"
if [ "${#experiment_dirs[@]}" -eq 0 ]; then
    echo "(none found)"
else
    printf '%s\n' "${experiment_dirs[@]}"
fi
echo "--- WandB offline run dir(s) ---"
if [ "${#wandb_dirs[@]}" -eq 0 ]; then
    echo "(none found)"
else
    printf '%s\n' "${wandb_dirs[@]}"
fi
echo "===================================="

if ! $DELETE; then
    echo
    echo "Dry run only - nothing deleted. Re-run with --delete to actually remove these."
    exit 0
fi

if [ ! -d "$slurm_dir" ] && [ "${#experiment_dirs[@]}" -eq 0 ] && [ "${#wandb_dirs[@]}" -eq 0 ]; then
    echo "Nothing found to delete."
    exit 0
fi

if ! $ASSUME_YES; then
    read -r -p "Delete all paths listed above? Type 'yes' to confirm: " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted - nothing deleted."
        exit 1
    fi
fi

[ -d "$slurm_dir" ] && rm -rf -- "$slurm_dir"
for d in "${experiment_dirs[@]}"; do
    rm -rf -- "$d"
done
for d in "${wandb_dirs[@]}"; do
    rm -rf -- "$d"
done

echo "Done."
