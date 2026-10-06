# JEPA on the NVIDIA Jetson Orin Nano Super

JEPA was run on the Jetson to compare it with RT-DETR, and then removed because RT-DETR is much faster there. This folder keeps what is needed to run it again. The platform setup (JetPack, power mode, PyTorch install) is the same as in `/RT-DETR/Jetson-Orin-Nano-Super/README.md`; skip its CUDA-graph patch, which JEPA does not use.

## Files

`jepa_server.py` is a small FastAPI wrapper (no Triton) around the **same** `model.py` and heads as the RTX box. It stubs the `triton_python_backend_utils` import and serves the same KServe v2 protocol on port 8000, model `vjepa_inspect`, so the phone app works unchanged.

## Run

1. Copy `model.py`, `plate_logic.py`, `obj_logic.py`, `plate_head.pt`, `brake_head.pt` and `battery_head.pt` from `/JEPA/RTX-PRO-4000/triton_model_repo/vjepa_inspect/1/` into the same folder as `jepa_server.py`.
2. `LD_LIBRARY_PATH=/usr/local/cuda/lib64 python -m uvicorn jepa_server:app --host 0.0.0.0 --port 8000`
   The V-JEPA 2 encoder (`facebook/vjepa2-vitl-fpc16-256-ssv2`, about 1.3 GB) is downloaded from Hugging Face on first start. Allow about a minute to load.
3. Do not run it together with the RT-DETR server on an 8 GB unit unless you have checked memory: JEPA alone used about 3.7 GB.

## Measured on the Jetson (25 W mode, then 10-phone sweep in MAXN Super)

| Frames per request | Server time |
|---|---|
| 4 to 8 | about 370 ms |
| 16 | about 445 ms |

With 16 frames of 256 px per request and one request per second per phone, the Jetson saturated at about 2.6 requests per second: median round trip 510 ms with 1 phone, 1.4 s with 3 phones and 4.7 s with 10 phones (`/RT-DETR/eval/jetson/sweep_jepa.json`). Answers matched the RTX box on the real battery photos tested. JEPA ran as plain PyTorch; no CUDA graphs or other optimisation was tried for it.
