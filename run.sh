#!/bin/sh

export PYTHONPATH="src:/../refer"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
export CUDA_VISIBLE_DEVICES="0,1"

python main.py $1
#kernprof -l main.py $1
