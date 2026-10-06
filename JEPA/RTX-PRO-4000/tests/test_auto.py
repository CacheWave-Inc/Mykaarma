"""Auto mode (task byte 3) through the live endpoint: does it pick the right item? usage: test_auto.py [triton|rtdetr] [n]"""
import collections
import json
import struct
import sys
import urllib.request

import cv2
import numpy as np

import synth_objects as SO
import synth_plates as SP

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "triton"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
URL = {"triton": "http://localhost:8000/v2/models/vjepa_inspect/infer", "rtdetr": "http://localhost:8100/v2/models/rtdetr_inspect/infer"}[ENGINE]


def call(task, frames):
    body = bytes([task, len(frames)])
    for f in frames:
        ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        b = enc.tobytes()
        body += struct.pack(">I", len(b)) + b
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}],
                      "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    req = urllib.request.Request(URL, data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    r = urllib.request.urlopen(req, timeout=30)
    hl = int(r.headers["Inference-Header-Content-Length"])
    return np.frombuffer(r.read()[hl:], dtype="<f4")


GEN = {0: lambda s: SO.make_clip("brake", s), 1: lambda s: SO.make_clip("battery", s), 2: SP.make_clip}
NAME = {0: "brake", 1: "battery", 2: "plate", -1: "nothing"}
for t in (0, 1, 2):
    conf = collections.Counter(); ok_codes = collections.Counter()
    for seed in range(400000, 400000 + N):
        frames, lab = GEN[t](seed)
        if not (lab["present"] and lab["done"]):
            continue
        v = call(3, frames)
        conf[NAME[int(v[9])]] += 1
        if int(v[9]) == t:
            ok_codes[int(v[0]) == 0] += 1
    tot = sum(conf.values())
    print(f"{NAME[t]:8s} good clips={tot:3d}  detected as: {dict(conf)}   correct task & OK code: {ok_codes[True]}/{tot}")
# clips of a plate scene WITHOUT a plate must not be taken for a battery or a wheel
conf = collections.Counter()
for seed in range(400000, 400000 + N):
    frames, lab = SP.make_clip(seed)
    if lab["present"] == 0:
        conf[NAME[int(call(3, frames)[9])]] += 1
print("plate-scene negatives (no target) detected as:", dict(conf))
