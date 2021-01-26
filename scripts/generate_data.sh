#!/bin/bash
# This script generates training data from a given directory by looking 
# for all the files in that directory that end wih ".mid" and converting
# them to a five track pianoroll dataset.
# Usage: ./generate_data.sh [DATA_DIR]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python "$DIR/../src/generate_data.py" $1
