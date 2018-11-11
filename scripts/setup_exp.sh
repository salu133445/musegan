#!/bin/bash
# This script set ups a new exeperiment by creating an experiment directory with
# a default configuration file and a default model parameter file.
# Usage: ./setup_exp.sh [EXP_DIR] [EXP_NOTE]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -d "$1" ]; then
  mkdir -p "$1"
fi
cp "$DIR/../src/musegan/default_params.yaml" "$1/params.yaml"
cp "$DIR/../src/musegan/default_config.yaml" "$1/config.yaml"
if [ ! -z "$2" ]; then
  echo "$2" > "$1/exp_note.txt"
fi
