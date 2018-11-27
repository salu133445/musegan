#!/bin/bash
# This script performs inference from a trained model. It first looks for the
# configuration and model parameter files in the experiment directory and then
# performs inference from the trained model on the specified GPU.
# Usage: run_inference.sh [EXP_DIR] [GPU_NUM]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -f "$1/config.yaml" ]; then
  echo "Configuration file not found" && exit 1
fi
if [ ! -f "$1/params.yaml" ]; then
  echo "Model parameter file not found" && exit 1
fi
if [ ! -d "$1/model" ]; then
  echo "Model checkpoint directory not found" && exit 1
fi
if [ -z "$2" ]; then
  gpu="0"
else
  gpu="$2"
fi
python3 "$DIR/../src/inference.py" --checkpoint_dir "$1/model" \
  --result_dir "$1/results/inference" --params "$1/params.yaml" \
  --config "$1/config.yaml" --runs 10 --gpu "$gpu"
