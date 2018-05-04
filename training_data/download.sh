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
    fileid=1F7J5n9uOPqViBYpoPT5GvE4PjCWhOyWc
    ;;
  "lastfm_alternative_8b_phrase.npy")
    fileid=1x3CeSqE6ElWa6V7ueNl8FKPFmMoyu4ED
    ;;
  *)
    echo "File not found"
    exit 1
    ;;
esac

wget -O $filename --no-check-certificate \
  "https://drive.google.com/uc?export=download&id="$fileid
