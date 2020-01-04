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
        fileid=1-KgUFsVvRXSWtpiB5zrRjTvFnss-6E5T
        ;;
      "lastfm_alternative_g_jamming_d_proposed.tar.gz")
        fileid=1D0vwE-DaPafRd5HM849Qbs3VmqSZI5I3
        ;;
      "lastfm_alternative_g_hybrid_d_proposed.tar.gz")
        fileid=1mFG3LwazgQ3YgoXoEm_blta8p5Gf2LaR
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
        fileid=16LKWjiEjDjgiTjMLFcgnzZdCT-v8fp3T
        ;;
      "lastfm_alternative_first_stage_d_ablated.tar.gz")
        fileid=1YyKAiPV0AuGuQB1K05dQAkPnsRMAqjtJ
        ;;
      "lastfm_alternative_first_stage_d_baseline.tar.gz")
        fileid=1ZVASqhTApVWSvtM0N-952BAEbfUqRTfK
        ;;
      *)
        echo "File not found"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unrecognizable model name"
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
