#!/bin/bash
sh download.sh musegan lastfm_alternative_g_hybrid_d_proposed

for postfix in proposed ablated baseline
do
  sh download.sh bmusegan lastfm_alternative_first_stage_d_$postfix
done
