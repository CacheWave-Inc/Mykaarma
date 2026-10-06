# RT-DETR on the RTX PRO 4000 box

Runs in Docker on port 8100, next to the JEPA/Triton container (port 8000). The container reuses the host Python environment (torch and transformers) read-only because the box has little disk space.

`run_rtdetr.sh` (as deployed) starts it:
- image `nvcr.io/nvidia/tritonserver:25.09-py3`, used only as a CUDA base,
- mounts the model folder (`MODEL_DIR`), the host `uv` Python tree and the server folder,
- runs `python -m uvicorn rtdetr_server:app --host 0.0.0.0 --port 8100`.

Adjust the paths inside `run_rtdetr.sh` to where you put `RT-DETR/models` and `RT-DETR/server`. `Dockerfile` builds a self-contained image instead (`docker build -t cachewave/rtdetr-server:1 .`); it was not built on this box because of disk space.

CUDA graphs (`USE_CUDA_GRAPH=1`) are **off** here. They need a small patch to the installed `transformers` (see `../Jetson-Orin-Nano-Super/`), and the RTX box does not need them: any item takes about 40 ms and one task about 15 ms per request.

mDNS discovery: the app looks for `_cachewave-triton._tcp`. The RTX box advertises both engines: JEPA on port 8000 (`/JEPA/RTX-PRO-4000/deploy/cachewave-triton.avahi.service`) and RT-DETR on port 8100 (`cachewave-rtdetr.avahi.service` in this folder). Copy them to `/etc/avahi/services/`.
