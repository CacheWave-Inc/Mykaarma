"""CacheWave RT-DETR inference server. Speaks the same KServe-v2 binary wire format as the Triton `vjepa_inspect` model, so the
Android app switches engines by changing only the address (port 8100) and model name (`rtdetr_inspect`).

POST /v2/models/rtdetr_inspect/infer
  request : [JSON header][FRAMES uint8 bytes], header `Inference-Header-Content-Length`
            FRAMES = [task:1 (bit 7 = verify a captured still, more tolerant)][n:1] + n x ([len:4 BE][JPEG, any square size; 256 for the old phone app, 512 for RT-DETR mode])
            task 0 = brake, 1 = battery, 2 = license plate
  response: [JSON header][12 x float32 LE] = [code, ready, dx, dy, size, brightness, sharpness, motion, present_prob, done_prob,
            infer_ms, frames_used]   (same layout as vjepa_inspect; present_prob is normalised so >= 0.85 means 'present')
Detection is single-frame (last frame); motion is measured over the last frames."""
import json
import os
import struct
import threading
import time
import types
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from transformers import RTDetrForObjectDetection

import det_logic as DL

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
MODEL_NAME = "rtdetr_inspect"
AUTO = 3          # task byte 3: look for any of the three items; v[9] then carries the detected task (-1 = nothing found)
TASKS = {0: "brake", 1: "battery", 2: "plate"}
R = 320
MOTION_MAX = 12.0
SHARP_MIN = {"brake": 15.0, "battery": 15.0, "plate": 20.0}
DARK_MIN = {"brake": 35, "battery": 35, "plate": 45}
THR = {"brake": 0.5, "battery": 0.5, "plate": 0.5}
VERIFY_MARGIN = -0.10

USE_GRAPH = os.environ.get("USE_CUDA_GRAPH", "0") == "1"      # replay each detector as one recorded CUDA graph (about 3.6x faster on the Jetson, whose slow CPU limits plain PyTorch)


class GraphDetector:
    """Records one forward pass of a detector as a CUDA graph and replays it. Same call/answer shape as the Hugging Face model.
    Needs the spatial_shapes cache patch in transformers' modeling_rt_detr.py (no host->device copies inside the forward)."""

    def __init__(self, model, res):
        self.model = model
        self.static = torch.zeros(1, 3, res, res, device="cuda", dtype=torch.float16)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._run()

    def _run(self):
        o = self.model(pixel_values=self.static)
        return o.logits, o.pred_boxes

    def __call__(self, pixel_values):
        self.static.copy_(pixel_values)
        self.graph.replay()
        return types.SimpleNamespace(logits=self.out[0], pred_boxes=self.out[1])


models = {}
model_cov = None            # detector for 'battery with its cover on' (battery task only)
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    thr_file = os.path.join(MODEL_DIR, "rtdetr_thresholds.json")
    if os.path.exists(thr_file):
        THR.update(json.load(open(thr_file)))
    global model_cov

    def load(dirname):
        path = os.path.join(MODEL_DIR, dirname)
        if not os.path.isdir(path):
            print(f"[rtdetr] {dirname} not found, skipped", flush=True)
            return None
        m = RTDetrForObjectDetection.from_pretrained(path, dtype=torch.float16).cuda().eval()
        res_file = os.path.join(path, "cw_res.json")
        res = json.load(open(res_file))["res"] if os.path.exists(res_file) else R      # each model remembers its training resolution
        with torch.no_grad():
            for _ in range(3):
                m(pixel_values=torch.rand(1, 3, res, res, device="cuda", dtype=torch.float16))
            if USE_GRAPH:
                m = GraphDetector(m, res)
        return m, res

    for t, name in TASKS.items():
        m = load(f"rtdetr_{name}")
        if m is not None:
            models[t] = m           # (model, resolution)
    model_cov = load("rtdetr_battery_covered")
    torch.cuda.synchronize()
    print("[rtdetr] models loaded:", [TASKS[t] for t in models], "covered:", model_cov is not None, "thresholds", THR, flush=True)
    yield


app = FastAPI(lifespan=lifespan)


def parse(data):
    n = data[1]
    off, frames = 2, []
    for _ in range(n):
        (length,) = struct.unpack(">I", data[off:off + 4])
        off += 4
        img = cv2.imdecode(np.frombuffer(data[off:off + length], np.uint8), cv2.IMREAD_COLOR)
        off += length
        if img is not None:
            frames.append(img)
    return frames


def present_prob(score, thr):
    """Map the detector score so that 0.85 == the operating threshold (the app's probe test uses 0.85)."""
    if score >= thr:
        return 0.85 + 0.15 * (score - thr) / max(1e-6, 1 - thr)
    return 0.84 * score / max(1e-6, thr)


def score_request(payload: bytes):
    out = np.zeros(12, dtype=np.float32)
    verify = bool(payload[0] & 0x80)
    task = (payload[0] & 0x7F) if ((payload[0] & 0x7F) in TASKS or (payload[0] & 0x7F) == AUTO) else 2
    if task != AUTO and task not in models:
        out[0] = DL.ERROR
        return out
    frames = parse(payload)
    if not frames:
        out[0] = DL.ERROR
        return out
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    g256 = cv2.resize(gray[-1], (256, 256), interpolation=cv2.INTER_AREA) if gray[-1].shape[0] != 256 else gray[-1]
    brightness = float(g256.mean())
    last = [cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) for g in gray[-4:]]
    motion = float(np.mean([np.abs(last[i + 1] - last[i]).mean() for i in range(len(last) - 1)])) if len(last) > 1 else 0.0

    inputs = {}

    def tensor(res):
        if res not in inputs:
            src = frames[-1]
            rgb = cv2.cvtColor(cv2.resize(src, (res, res), interpolation=cv2.INTER_AREA if src.shape[0] > res else cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            inputs[res] = torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0).half() / 255
        return inputs[res]

    results = {}
    with lock, torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        for t in ([k for k in TASKS if k in models] if task == AUTO else [task]):
            mt, rt = models[t]
            o = mt(pixel_values=tensor(rt))
            sc = o.logits.float().sigmoid()[0]                   # [Q, K]
            boxes = o.pred_boxes.float()[0]                      # [Q, 4] cx, cy, w, h
            best = sc.max(0)
            qi = best.indices.cpu().numpy()
            results[t] = (best.values.cpu().numpy(), boxes[torch.as_tensor(qi, device=boxes.device)].cpu().numpy())
        cov_val = 0.0
        if task == 1 and model_cov is not None:                  # battery task chosen explicitly: is the cover still on?
            cov_val = float(model_cov[0](pixel_values=tensor(model_cov[1])).logits.float().sigmoid()[0].max())
        torch.cuda.synchronize()
        infer_ms = (time.time() - t0) * 1000.0

    detected = -1
    if task == AUTO:
        pick, pick_r = None, 1.0
        for t, (scr, _) in results.items():
            r = float(scr[0]) / THR[TASKS[t]]
            if r >= pick_r:
                pick, pick_r = t, r
        if pick is None:                                         # nothing recognisable in view
            sharp = float(cv2.Laplacian(g256, cv2.CV_64F).var())
            code = DL.DARK if brightness < 35 else DL.BRIGHT if brightness > 225 else DL.NO_PLATE
            out[:] = [code, 0.0, 0, 0, 0, brightness, sharp, motion, 0.0, -1.0, infer_ms, len(frames)]
            return out
        task = detected = pick
    name = TASKS[task]
    score, bsel = results[task]

    def xyxy(b):
        cx, cy, w, h = [float(v) for v in b]
        return (min(max(cx - w / 2, 0.0), 1.0), min(max(cy - h / 2, 0.0), 1.0), min(max(cx + w / 2, 0.0), 1.0), min(max(cy + h / 2, 0.0), 1.0))

    thr = THR[name]
    box = xyxy(bsel[0])
    cov = cov_val
    code = DL.decide(name, float(score[0]), box, cov, thr, margin=VERIFY_MARGIN if verify else DL.MARGIN)

    found = code not in (DL.NO_PLATE, DL.COVERED)
    x0, y0, x1, y1 = [int(v * 256) for v in box]
    has_region = found and x1 - x0 > 8 and y1 - y0 > 8
    region = g256[y0:y1, x0:x1] if has_region else g256
    sharp = float(cv2.Laplacian(region, cv2.CV_64F).var())
    light = float(g256[y0:y1, x0:x1].mean()) if (has_region and name == "plate") else brightness
    if light < DARK_MIN[name]:
        code = DL.DARK
    elif light > 225:
        code = DL.BRIGHT
    elif code == DL.OK and motion > MOTION_MAX and not verify:
        code = DL.SHAKY
    elif code == DL.OK and sharp < SHARP_MIN[name] * (0.5 if verify else 1.0):
        code = DL.BLURRY

    w, h = box[2] - box[0], box[3] - box[1]
    size = (w if name == "plate" else max(w, h)) if found else 0.0
    dx, dy = ((box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5) if found else (0.0, 0.0)
    out[:] = [code, 1.0 if code == DL.OK else 0.0, dx, dy, size, brightness, sharp, motion,
              present_prob(float(score[0]), thr), float(detected) if detected >= 0 else float(score[0]), infer_ms, len(frames)]
    return out


@app.get("/v2/health/ready")
@app.get("/v2/health/live")
def health():
    return Response(status_code=200 if models else 503)


@app.get("/v2")
def server_meta():
    return {"name": "cachewave-rtdetr", "version": "1", "engine": "rtdetr", "models": [MODEL_NAME]}


@app.get("/v2/models/{name}/ready")
def model_ready(name: str):
    return Response(status_code=200 if name == MODEL_NAME and models else 404)


@app.post("/v2/models/{name}/infer")
async def infer(name: str, request: Request):
    if name != MODEL_NAME or not models:
        return Response(status_code=404)
    body = await request.body()
    hl = int(request.headers.get("inference-header-content-length", "0"))
    payload = body[hl:]
    t_in = time.perf_counter()                       # request received (body in memory) -> answer ready to send: queueing + decode + inference + post-processing, no network
    out = await run_in_threadpool(score_request, payload)
    out[10] = (time.perf_counter() - t_in) * 1000.0  # slot 10 now carries the whole server time (it used to be the detector forward passes only)
    hdr = json.dumps({"model_name": MODEL_NAME, "outputs": [{"name": "GUIDANCE", "datatype": "FP32", "shape": [12],
                                                             "parameters": {"binary_data_size": 48}}]}).encode()
    return Response(content=hdr + out.astype("<f4").tobytes(), media_type="application/octet-stream",
                    headers={"Inference-Header-Content-Length": str(len(hdr))})
