#!/bin/bash
# This script reruns an experiment (training, inference and interpolation). It
# will remove everything in the experiment directory except the configuration
# and model parameter files and then reruns the experiment on the specified GPU.
# Usage: ./rerun_exp.sh [EXP_DIR] [GPU_NUM]
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
find "$1" -mindepth 1 -maxdepth 1 -not -name "params.yaml" \
  -not -name "config.yaml" -not -name "exp_note.txt" -exec rm -r {} +
"$DIR/run_exp.sh" "$1" "$gpu"
