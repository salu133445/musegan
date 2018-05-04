#!/bin/bash
case $2 in
  *.tar.gz)
    filename=$2
    ;;
  *)
    filename=$2.tar.gz
    ;;
esac

case $1 in
  "musegan"|"MuseGAN")
    case $filename in
      "lastfm_alternative_g_hybrid_d_proposed.tar.gz")
        fileid=1b1bwTzW09QPFbRn2Hy9X8yU1fbTc3S1k
        ;;
      *)
        echo "File not found"
        exit 1
        ;;
    esac
  "bmusegan"|"binarymusegan"|"BinaryMuseGAN")
    case $filename in
      "lastfm_alternative_first_stage_d_proposed.tar.gz")
        fileid=12tEzs-Qa-qi59hLJB8TlD-vcZgVEQZu6
        ;;
      "lastfm_alternative_first_stage_d_ablated.tar.gz")
        fileid=1GolkoE2ktmHF2Pt7POd8TBBYZARu6ih8
        ;;
      "lastfm_alternative_first_stage_d_baseline.tar.gz")
        fileid=1qWWWU6UTMJvzdK6y4bvh3PRXF5Xbk09v
        ;;
      *)
        echo "File not found"
        exit 1
        ;;
    esac
  *)
    echo "Unrecognizeable model name"
    exit 1
    ;;
esac

wget -O $filename --no-check-certificate \
  "https://drive.google.com/uc?export=download&id="$fileid
