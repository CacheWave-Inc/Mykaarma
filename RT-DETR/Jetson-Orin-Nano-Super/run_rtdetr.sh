#!/bin/bash
# Starts the RT-DETR server on the Jetson (port 8100). Paths below are the ones used on the test unit; change them if yours differ.
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export USE_CUDA_GRAPH=1 MODEL_DIR="$HOME/rtdetr/models" HF_HUB_OFFLINE=1
cd "$HOME/rtdetr"
exec "${VENV:-$HOME/jepa/venv}/bin/python" -W ignore -m uvicorn rtdetr_server:app --host 0.0.0.0 --port 8100
