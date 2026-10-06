# CacheWave Inspector: RT-DETR edge node

A small FastAPI server (`server/rtdetr_server.py`) that guides a technician to photograph a **license plate, battery and brake**, using one fine-tuned RT-DETR (ResNet-18) detector per item. It speaks the same KServe v2 binary protocol as the JEPA/Triton node (`/JEPA`), so the Android app (`/app`) switches engines by changing only the address and model name.

| | RT-DETR node | JEPA node |
|---|---|---|
| Port / model name | 8100 / `rtdetr_inspect` | 8000 / `vjepa_inspect` |
| Input | last frame, any square JPEG (the app sends 512 px) | last 16 frames at 256 px |
| Runs on | RTX PRO 4000 (Docker) and Jetson Orin Nano Super (systemd) | RTX PRO 4000 (Triton) |

Platform setup: `RTX-PRO-4000/` and `Jetson-Orin-Nano-Super/`.

## Layout

| Path | What |
|---|---|
| `server/rtdetr_server.py`, `server/det_logic.py` | The server and the rules that turn a detection into guidance |
| `models/rtdetr_{plate,brake,battery}/` | The three deployed detectors (Hugging Face `RTDetrForObjectDetection` format, **fp16**) |
| `training/` | `train_det.py` (fine-tuning, optional real-frame mixing), run scripts, `label_owl.py` (OWLv2 auto-labelling of real frames) |
| `tests/` | End-to-end test, folder test, load test, phone emulators, CUDA-graph benchmark |
| `eval/` | Held-out metrics, training logs, benchmarks; `eval/jetson/` has the 1-to-10-phone results |

## Models

Each detector is `PekingU/rtdetr_r18vd` fine-tuned on synthetic scenes (generators in `/JEPA/RTX-PRO-4000/training/synth_*.py`). Plate and brake run at 320 px, battery at 448 px (`cw_res.json` in the model folder; the server reads it).

The weights here are stored as **fp16** (about 38 MB each instead of 77 MB) so they fit in plain Git. The server loads the models as fp16 anyway; outputs were checked to be bit-identical to the original fp32 files. The fp32 originals are not in this repo, so fine-tuning from these files starts from fp16 values.

Not included: the fine-tune on real phone frames (`rtdetr_battery_real`, not evaluated), the covered-battery detector (never learned), and the training data.

## Server

`POST /v2/models/rtdetr_inspect/infer`, body = JSON header + `FRAMES` bytes, header length in `Inference-Header-Content-Length`.

`FRAMES` = `[task:1][n:1]` + n x `([len:4 big-endian][JPEG, square])`. `task`: 0 brake, 1 battery, 2 plate, 3 auto (all three detectors, best one wins). Bit 7 = verify a still photo (looser margin).

Answer = 12 float32: `[code, ready, dx, dy, size, brightness, sharpness, motion, present_prob, done_prob, server_ms, frames_used]`. Codes are listed in `/JEPA/RTX-PRO-4000/README.md`. In auto mode slot 9 holds the detected task (-1 = nothing). `server_ms` (slot 10) is the whole handling time from request received to answer ready, including waiting for the GPU.

Environment variables: `MODEL_DIR` (folder holding the `rtdetr_*` model folders; optional `rtdetr_thresholds.json`), `USE_CUDA_GRAPH=1` (see Jetson), `HF_HUB_OFFLINE=1`.

## Measured performance (5 October 2026)

Jetson Orin Nano Super, MAXN Super mode, CUDA graphs on. Each phone sends one request per second (2 frames of 512 px). Median round trip from a PC on the same Wi-Fi; raw data in `eval/jetson/`.

| Phones | Any item (3 detectors) | One task | JEPA (for comparison, plain PyTorch, 16 frames) |
|---|---|---|---|
| 1 | 91 ms | 51 ms | 510 ms |
| 5 | 95 ms | 54 ms | 2,327 ms |
| 10 | 112 ms | 57 ms | 4,665 ms |

Throughput ceiling: about 20 requests/s any item and 46 requests/s one task on the Jetson; about 2.6 requests/s for JEPA.

The scripts in `tests/` read saved frames from `$FRAMES_DIR` (a folder of 512 px or larger JPEGs) and photos from `$PHOTO_DIR`, and address the node at `192.168.68.63:8100` (edit `BASE` / the URL at the top). The frames and photos themselves are not in this repo.

## Known limits

- The detectors were trained on **synthetic** scenes. On three real battery photos tested, RT-DETR found none (JEPA found two). Plate is also weaker on real photos than on synthetic ones. Mixing real frames into training (`REAL_DIR`/`REAL_JSON` in `train_det.py`) was started but not finished.
- These numbers measure speed, not accuracy.
