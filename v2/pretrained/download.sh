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
      "lastfm_alternative_g_composer_d_proposed.tar.gz")
        fileid=1QzTL4So-oRWrif4gVKqQM5yQ48y2X5gM
        ;;
      "lastfm_alternative_g_jamming_d_proposed.tar.gz")
        fileid=1-Q_krj4VKOWbpFU1jTdfihKYxV6hKmlm
        ;;
      "lastfm_alternative_g_hybrid_d_proposed.tar.gz")
        fileid=1b1bwTzW09QPFbRn2Hy9X8yU1fbTc3S1k
        ;;
      *)
        echo "File not found"
        exit 1
        ;;
    esac
    ;;
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
    ;;
  *)
    echo "Unrecognizeable model name"
    exit 1
    ;;
esac

confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
  --no-check-certificate \
  "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn \
  's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget -O "$filename" --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$confirm&id=$fileid"
rm -rf /tmp/cookies.txt
