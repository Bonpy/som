#!/bin/bash

COLOURS=${COLOURS:-10}
ITERATIONS=${ITERATIONS:-100}
OUTPUT=${OUTPUT:-output.png}
GRID_WIDTH=${GRID_WIDTH:-100}
GRID_HEIGHT=${GRID_HEIGHT:-100}
INPUT_DIM=${INPUT_DIM:-3}
LR=${LR:-0.1}

exec python ./scripts/train.py