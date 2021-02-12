#!/bin/bash
# This script downloads the pretrained models.
# Usage: download_models.sh
#
# Make sure `gdown` is installed, otherwise install it by `pip install gdown`.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
if [ ! -d "$DIR/../exp" ]; then
  mkdir "$DIR/../exp"
fi
gdown -O "$DIR/../exp/pretrained_models.tar.gz" \
  --id "19RYAbj_utCDMpU7PurkjsH4e_Vy8H-Uy"
echo "Decompressing archive."
if tar zxf "$DIR/../exp/pretrained_models.tar.gz" -C "$DIR/../exp"; then
    rm "$DIR/../exp/pretrained_models.tar.gz"
    echo "Successfully decompressed."
else
    echo "Decompression failed."
fi
