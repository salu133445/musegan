#!/bin/bash
# This script download the training data.
# Usage: download_data.sh
download () {
  confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
    --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$2" -O- | sed -rn \
    's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget -O "$1" --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$confirm&id=$2"
  rm -rf /tmp/cookies.txt
}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -d "$DIR/../data" ]; then
  mkdir "$DIR/../data"
fi
download "$DIR/../data/train_x_lpd_5_phr.npz" \
  "14rrC5bSQkB9VYWrvt2IhsCjOKYrguk3S"
