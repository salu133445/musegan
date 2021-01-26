#!/bin/bash
# This script downloads the training data.
# Usage: download_data.sh
#
# Make sure `gdown` is installed, otherwise install it by `pip install gdown`.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -d "$DIR/../data" ]; then
  mkdir "$DIR/../data"
fi
gdown -O "$DIR/../data/train_x_lpd_5_phr.npz" \
  --id "14rrC5bSQkB9VYWrvt2IhsCjOKYrguk3S"
