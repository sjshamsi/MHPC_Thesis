#!/bin/bash
# sbatch, but first creates the script's --output folder: SLURM won't, and without it
# the job's .out/.err are silently dropped.
#
# Usage (any sbatch options go before the script):
#   Tools/slurm/submit.sh Runs/test_flops/test_flops.sh
#   Tools/slurm/submit.sh --dependency=singleton --export=ALL,AUTO_RESUME=False <script.sh>

set -euo pipefail

script="${!#}"  # last argument
if [[ $# -eq 0 || ! -f "$script" ]]; then
    echo "usage: $0 [sbatch options] <script.sh>" >&2
    exit 1
fi

job_name=$(grep -m1 -oP '(?<=#SBATCH --job-name=)\S+' "$script" || true)
for out in $(grep -oP '#SBATCH --(output|error)=\K\S+' "$script" || true); do
    mkdir -p "$(dirname "${out//%x/$job_name}")"
done

exec sbatch "$@"
