#!/bin/bash
# This script download the pretrained models.
# Usage: download_models.sh
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
if [ ! -d "$DIR/../exp" ]; then
  mkdir "$DIR/../exp"
fi
download "$DIR/../exp/pretrained_models.tar.gz" \
  "19RYAbj_utCDMpU7PurkjsH4e_Vy8H-Uy"
echo "Decompressing archive."
if tar zxf "$DIR/../exp/pretrained_models.tar.gz" -C "$DIR/../exp"; then
    rm "$DIR/../exp/pretrained_models.tar.gz"
    echo "Successfully decompressed."
else
    echo "Decompression failed."
fi
