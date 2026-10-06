FROM nvidia/cuda:12.6.0-base-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# torch cu128 bundles its own CUDA runtime, so only the host driver is needed
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir "transformers==5.17.0" fastapi uvicorn opencv-python-headless numpy pillow

WORKDIR /app
COPY det_logic.py rtdetr_server.py /app/
ENV MODEL_DIR=/models
EXPOSE 8100
CMD ["uvicorn", "rtdetr_server:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
