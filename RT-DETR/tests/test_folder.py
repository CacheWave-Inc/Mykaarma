"""Test every photo in battery_photos with both engines (battery mode and auto mode), centre-square crop like the app."""
import glob
import io
import json
import os
import struct
import sys
import urllib.request

import numpy as np
from PIL import Image

FOLDER = os.environ.get("PHOTO_DIR", "photos")
CODE = "OK DARK BRIGHT BLURRY SHAKY LEFT RIGHT UP DOWN CLOSER BACK WARMING ERROR NO_TARGET ADJUST COVERED".split()
NAME = {0: "brake", 1: "battery", 2: "plate", -1: "nothing"}


def jpg(c):
    b = io.BytesIO(); c.save(b, "JPEG", quality=85); return b.getvalue()


def call(base, model, task, jpgs):
    body = bytes([task, len(jpgs)]) + b"".join(struct.pack(">I", len(j)) + j for j in jpgs)
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}],
                      "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    req = urllib.request.Request(f"{base}/v2/models/{model}/infer", data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    r = urllib.request.urlopen(req, timeout=30); hl = int(r.headers["Inference-Header-Content-Length"])
    return np.frombuffer(r.read()[hl:], dtype="<f4")


print(f"{'photo':22s}{'size':>11s} | {'RT-DETR battery':26s}| {'RT-DETR auto':22s}| {'JEPA battery':26s}| JEPA auto")
for f in sorted(glob.glob(os.path.join(FOLDER, "*"))):
    if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue
    im = Image.open(f).convert("RGB")
    w, h = im.size
    s = min(w, h); x0, y0 = (w - s) // 2, (h - s) // 2
    sq = im.crop((x0, y0, x0 + s, y0 + s))
    c512, c256 = sq.resize((512, 512), Image.LANCZOS), sq.resize((256, 256), Image.LANCZOS)
    out = []
    for base, model, jp in (("http://192.168.68.63:8100", "rtdetr_inspect", [jpg(c512)]), ("http://192.168.68.63:8000", "vjepa_inspect", [jpg(c256)] * 4)):
        b = call(base, model, 1, jp)
        a = call(base, model, 3, jp)
        out.append(f"{CODE[int(b[0])]:9s} p={b[8]:.2f} sz={b[4]:.2f}")
        out.append(f"{NAME[int(a[9])]:8s} p={a[8]:.2f}")
    print(f"{os.path.basename(f):22s}{w:5d}x{h:<5d} | {out[0]:26s}| {out[1]:22s}| {out[2]:26s}| {out[3]}")
