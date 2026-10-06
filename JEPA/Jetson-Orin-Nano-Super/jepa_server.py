"""CacheWave JEPA inference server for the Jetson: runs the SAME model.py / heads as the Triton `vjepa_inspect` model, behind the same
KServe-v2 binary wire format (POST /v2/models/vjepa_inspect/infer on port 8000), so the phone app needs no change."""
import json
import os
import sys
import threading
import types
from contextlib import asynccontextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.modules["triton_python_backend_utils"] = types.ModuleType("triton_python_backend_utils")      # model.py imports it; the scoring code does not use it

import numpy as np  # noqa: E402
from fastapi import FastAPI, Request, Response  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402

import model as M  # noqa: E402
import plate_logic as PL  # noqa: E402

MODEL_NAME = "vjepa_inspect"
impl = None
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global impl
    impl = M.TritonPythonModel()
    impl.initialize(None)
    yield


app = FastAPI(lifespan=lifespan)


def score_request(payload: bytes):
    try:
        verify = bool(payload[0] & 0x80)
        t = payload[0] & 0x7F
        task = t if (t in M.TASKS or t == M.AUTO) else 2
        with lock:
            return impl._score(task, M.parse(payload), verify)
    except Exception as exc:
        print(f"[jepa_server] error: {exc!r}", flush=True)
        out = np.zeros(12, dtype=np.float32)
        out[0] = PL.ERROR
        return out


@app.get("/v2/health/ready")
@app.get("/v2/health/live")
def health():
    return Response(status_code=200 if impl is not None else 503)


@app.get("/v2")
def server_meta():
    return {"name": "cachewave-jepa-jetson", "version": "1", "engine": "jepa", "models": [MODEL_NAME]}


@app.get("/v2/models/{name}/ready")
def model_ready(name: str):
    return Response(status_code=200 if name == MODEL_NAME and impl is not None else 404)


@app.post("/v2/models/{name}/infer")
async def infer(name: str, request: Request):
    if name != MODEL_NAME or impl is None:
        return Response(status_code=404)
    body = await request.body()
    hl = int(request.headers.get("inference-header-content-length", "0"))
    out = await run_in_threadpool(score_request, body[hl:])
    hdr = json.dumps({"model_name": MODEL_NAME, "outputs": [{"name": "GUIDANCE", "datatype": "FP32", "shape": [12],
                                                             "parameters": {"binary_data_size": 48}}]}).encode()
    return Response(content=hdr + out.astype("<f4").tobytes(), media_type="application/octet-stream",
                    headers={"Inference-Header-Content-Length": str(len(hdr))})
