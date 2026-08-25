#!/bin/bash
# Thin wrapper around `sbatch` for the hand-written scripts in this directory
# (generated star_search_run_N.sh / tuning_run_N.sh scripts don't need this -
# their generators already create their slurm/<job_name>/ directory).
#
# SLURM writes --output/--error to logs/slurm/<job_name>/<job_id>.out but will
# NOT create the <job_name> subdirectory itself, so submitting straight via
# `sbatch some_script.sh` fails/silently drops output the first time a given
# job name is used. This wrapper reads --job-name out of the script, makes
# sure its log directory exists, then forwards everything to sbatch.
#
# Usage:
#   ./submit.sh anchor_dimension_run.sh
#   ./submit.sh --export=ALL,AUTO_RESUME=False anchor_dimension_run.sh

set -euo pipefail

LOGS_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs

script=""
for arg in "$@"; do
    case "$arg" in
        -*) ;;
        *) script="$arg" ;;
    esac
done

if [[ -z "$script" || ! -f "$script" ]]; then
    echo "usage: $0 [sbatch options] <script.sh>" >&2
    exit 1
fi

job_name=$(grep -m1 -oP '(?<=#SBATCH --job-name=).+' "$script")
if [[ -z "$job_name" ]]; then
    echo "Could not find #SBATCH --job-name= in $script" >&2
    exit 1
fi

mkdir -p "$LOGS_DIR/slurm/$job_name"
exec sbatch "$@"
