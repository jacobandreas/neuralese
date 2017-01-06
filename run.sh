#!/bin/sh

export PYTHONPATH="src:/../refer"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

python main.py
