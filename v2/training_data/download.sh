#!/bin/bash
case $1 in
  *.npy)
    filename=$1
    ;;
  *)
    filename=$1.npy
    ;;
esac

case $filename in
  "lastfm_alternative_5b_phrase.npy")
    fileid=1QKKWJ9t7K8nwYRD_AbNQDUie4xSiv1ph
    ;;
  "lastfm_alternative_8b_phrase.npy")
    fileid=1f9NKbhIxIbedHR370sc_hF9730985Xre
    ;;
  *)
    echo "File not found"
    exit 1
    ;;
esac

confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
  --no-check-certificate \
  "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn \
  's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget -O $filename --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$confirm&id=$fileid"
rm -rf /tmp/cookies.txt
