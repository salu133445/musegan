#!/bin/bash
# This script performs interpolation on the latent space of a trained model. It
# first looks for the configuration and model parameter files in the experiment
# directory and then performs interpolation on the latent space of the trained
# model on the specified GPU.
# Usage: run_interpolation.sh [EXP_DIR] [GPU_NUM]
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
python3 "$DIR/../src/interpolation.py" --checkpoint_dir "$1/model" \
  --result_dir "$1/results/interpolation" --params "$1/params.yaml" \
  --config "$1/config.yaml" --lower 0.0 --upper 1.0 --runs 10 --gpu "$gpu"
