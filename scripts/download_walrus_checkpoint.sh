#!/bin/bash
# Download the released pretrained Walrus checkpoint + its extended config from HuggingFace.
# Mirrors demo_notebooks/walrus_example_1_RunningWalrus.ipynb.
set -euo pipefail

BASE_PATH="${1:-/leonardo_scratch/fast/ICT26_MHPC_0/sshamsi/checkpoints/walrus_pretrained}"

mkdir -p "$BASE_PATH"

wget https://huggingface.co/polymathic-ai/walrus/resolve/main/extended_config.yaml \
    -O "$BASE_PATH/extended_config.yaml"
wget https://huggingface.co/polymathic-ai/walrus/resolve/main/walrus.pt \
    -O "$BASE_PATH/walrus.pt"

echo "Downloaded checkpoint + config to $BASE_PATH"
