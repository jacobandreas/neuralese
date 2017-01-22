#!/bin/sh

export PYTHONPATH="src:/../refer"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
export CUDA_VISIBLE_DEVICES="2,3"

python src/server/server.py
