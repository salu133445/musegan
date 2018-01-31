#!/bin/bash
python converter.py lpd lpd-5 # create lpd_full, lpd_5_full
python collector.py lpd lpd-5 # create lpd_matched, lpd_5_matched
python cleanser.py lpd lpd-5 # create lpd_cleansed, lpd_5_cleansed
