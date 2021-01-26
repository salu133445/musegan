#!/bin/bash
# This script runs an experiment (training, inference and interpolation). It
# first looks for the configuration and model parameter files in the experiment
# directory and then runs the experiment on the specified GPU.
# Usage: ./run_exp.sh [EXP_DIR] [GPU_NUM]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -f "$1/params.yaml" ]; then
  echo "Model parameter file not found" && exit 1
fi
if [ ! -f "$1/config.yaml" ]; then
  echo "Configuration file not found" && exit 1
fi
if [ -z "$2" ]; then
  gpu="0"
else
  gpu="$2"
fi
"$DIR/run_train.sh" "$1" "$gpu"
"$DIR/run_inference.sh" "$1" "$gpu"
"$DIR/run_interpolation.sh" "$1" "$gpu"
