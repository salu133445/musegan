#!/bin/bash
# This script trains a model. It first looks for the configuration and model
# parameter files in the experiment directory and then trains the model on the
# specified GPU.
# Usage: run_train.sh [EXP_DIR] [GPU_NUM]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -f "$1/config.yaml" ]; then
  echo "Configuration file not found" && exit 1
fi
if [ ! -f "$1/params.yaml" ]; then
  echo "Model parameter file not found" && exit 1
fi
if [ -z "$2" ]; then
  gpu="0"
else
  gpu="$2"
fi
python3 "$DIR/../src/train.py" --exp_dir "$1" --params "$1/params.yaml" \
  --config "$1/config.yaml" --gpu "$gpu"
