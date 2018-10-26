#!/bin/bash
for model in composer jamming hybrid
do
  sh download.sh musegan lastfm_alternative_g_${model}_d_proposed
done

for postfix in proposed ablated baseline
do
  sh download.sh bmusegan lastfm_alternative_first_stage_d_$postfix
done
