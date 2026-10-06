# RT-DETR on the NVIDIA Jetson Orin Nano Super

Tested on: Jetson Orin Nano Engineering Reference Developer Kit **Super**, 8 GB, 116 GB NVMe, Wi-Fi, JetPack 7.2.1 (L4T R39.2.1, Ubuntu 24.04, CUDA 13.2), aarch64, Python 3.12.

## Setup

1. **JetPack and Python tooling**
   `sudo apt install nvidia-jetpack python3-pip python3-venv`
2. **Power mode.** The unit defaults to 25 W. Switch to the top mode (it persists across reboots):
   `sudo nvpmodel -m 2` (`MAXN_SUPER`). Optionally pin the clocks with `sudo jetson_clocks` (resets on reboot).
3. **Python environment** (`requirements.txt`):
   ```
   python3 -m venv ~/jepa/venv && . ~/jepa/venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   pip install transformers opencv-python-headless fastapi uvicorn numpy safetensors
   ```
   - The official CUDA 13.0 wheel works on the Orin (compute capability 8.7) through its sm_80 binaries; it prints a harmless "no published build supports this GPU" warning.
   - There is no Jetson AI Lab wheel index for JetPack 7. The `sbsa/cu130` wheel from that site does **not** work: it only contains sm_110 and sm_121 code.
   - If `import torch` fails on `libnvpl_*` or `libcudss.so.0`, install `libnvpl-blas0` and `libnvpl-lapack0` (NVIDIA CUDA `ubuntu2404/sbsa` apt repository) and `pip install nvidia-cudss-cu13`.
   - Run with `LD_LIBRARY_PATH=/usr/local/cuda/lib64` (the run script sets it).
4. **CUDA-graph patch.** Plain PyTorch is limited by the Jetson's slow CPU (the GPU was only 22 to 29% busy). The server can record each detector once and replay it (`USE_CUDA_GRAPH=1`), which cut each detector from about 52 ms to 14 ms (battery 19 ms) with identical outputs. Hugging Face's RT-DETR code copies small tensors host-to-device on every forward, which is not allowed while recording, so apply the patch:
   ```
   cd ~/jepa/venv/lib/python3.12/site-packages
   cp transformers/models/rt_detr/modeling_rt_detr.py transformers/models/rt_detr/modeling_rt_detr.py.orig
   patch transformers/models/rt_detr/modeling_rt_detr.py < /path/to/modeling_rt_detr_spatial_shapes.patch
   ```
   Written against `transformers` 5.18.0. Re-check it after upgrading.
5. **Files.** Create `~/rtdetr/` with `rtdetr_server.py` and `det_logic.py` from `../server/`, the three folders from `../models/` under `~/rtdetr/models/`, and `run_rtdetr.sh` from this folder (`chmod +x`).
6. **Service** (starts at boot): edit `User=` and the path in `cachewave-rtdetr.service`, then
   `sudo cp cachewave-rtdetr.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now cachewave-rtdetr`
7. **Check:** `curl localhost:8100/v2/health/ready` returns 200 after about 30 s (loading and recording the graphs).

## Results (MAXN Super, CUDA graphs)

Server time per request: about 56 ms any item (three detectors), about 25 ms one task. With ten phones sending one request per second: median round trip 112 ms any item, 57 ms one task. Full tables in `../README.md`; raw data in `../eval/jetson/`. Before the graph change, ten phones in any-item mode took about 1.6 s (MAXN Super) or 2.3 s (25 W).

## What did not work

- **TensorRT.** The detectors export to ONNX (opset 17) fine, but `trtexec` (TensorRT 10.16) fails to build an engine on every variant tried (plain and onnxsim-simplified ONNX, FP16 and FP32, optimisation levels 1 and 3, workspace 512 MiB to 7 GiB): "Could not find any implementation for node {ForeignNode[.../m/Gather_1]}", with the Myelin pass reporting "Exceeded mem budget". Do not spend time on it without a different export.

## Gotchas

- After a service restart the phone app may fail its saved-address check while the server is still loading, rediscover the RTX box by mDNS, and silently save that address. Check the app's saved RT-DETR address after any restart.
- Both engines advertised by an RTX box on the same network will appear in mDNS discovery; the app uses its saved address first.
