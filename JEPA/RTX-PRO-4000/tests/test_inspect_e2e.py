"""End-to-end through the live Triton HTTP endpoint with unseen seeds, for all three tasks (0 brake, 1 battery, 2 plate)."""
import collections
import json
import struct
import sys
import urllib.request

import cv2
import numpy as np

import plate_logic as PL
import synth_objects as SO
import synth_plates as SP

URL = "http://localhost:8000/v2/models/vjepa_inspect/infer"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


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
NAME = {0: "brake", 1: "battery", 2: "plate"}
for task in (0, 1, 2):
    tot = collections.Counter()
    for seed in range(300000, 300000 + N):
        frames, lab = GEN[task](seed)
        code = int(call(task, frames)[0])
        if lab.get("covered"):
            tot["n_cov"] += 1; tot["cov_flagged"] += code == 15; tot["cov_ok"] += code == PL.OK
        if lab["present"] == 0:
            tot["n_none"] += 1; tot["ok_none"] += code == PL.OK
        elif lab["done"] == 0:
            tot["n_bad"] += 1; tot["ok_bad"] += code == PL.OK
        else:
            tot["n_good"] += 1; tot["ok_good"] += code == PL.OK
    print(f"{NAME[task]:8s} OK with no target {tot['ok_none']}/{tot['n_none']}   OK on bad/incomplete {tot['ok_bad']}/{tot['n_bad']}   OK on good {tot['ok_good']}/{tot['n_good']}"
          + (f"   covered battery: flagged COVERED {tot['cov_flagged']}/{tot['n_cov']}, said OK {tot['cov_ok']}/{tot['n_cov']}" if tot["n_cov"] else ""))
