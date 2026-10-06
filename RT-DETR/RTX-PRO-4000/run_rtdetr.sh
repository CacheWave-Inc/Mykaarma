#!/bin/sh
# Starts the RT-DETR inference container on port 8100 (the JEPA/Triton container keeps port 8000).
# Dev mode: reuses the project Python environment (torch/transformers) read-only, because the box has no room for a full image.
# For a self-contained image use the Dockerfile in this folder: docker build -t cachewave/rtdetr-server:1 .
docker rm -f rtdetr 2>/dev/null
docker run -d --name rtdetr --restart unless-stopped --gpus all -p 8100:8100   -e MODEL_DIR=/home/cw-biz1/vjepa-bench -e HF_HUB_OFFLINE=1   -v /home/cw-biz1/vjepa-bench:/home/cw-biz1/vjepa-bench:ro   -v /home/cw-biz1/.local/share/uv:/home/cw-biz1/.local/share/uv:ro   -v /home/cw-biz1/rtdetr-server:/app:ro -w /app   nvcr.io/nvidia/tritonserver:25.09-py3   /home/cw-biz1/vjepa-bench/.venv/bin/python -m uvicorn rtdetr_server:app --host 0.0.0.0 --port 8100
