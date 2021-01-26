#!/bin/bash
# This script stores the training data to shared memory.
# Usage: process_data.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python "$DIR/../src/process_data.py" "$DIR/../data/train_x_lpd_5_phr.npz"
