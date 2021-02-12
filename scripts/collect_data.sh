#!/bin/bash
# This script collect training data from a given directory by looking
# for all the files in that directory that end wih ".mid" and converting
# them to a five-track pianoroll dataset.
# Usage: ./generate_data.sh [INPUT_DIR] [OUTPUT_FILENAME]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
python "$DIR/../src/collect_data.py" -i "$1" -o "$2"
