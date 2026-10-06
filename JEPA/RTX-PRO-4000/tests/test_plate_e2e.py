import json, struct, urllib.request, numpy as np, cv2, collections
from synth_plates import make_clip
import plate_logic as PL
URL = "http://localhost:8000/v2/models/vjepa_plate/infer"

def call(frames):
    body = bytes([2, len(frames)])
    for f in frames:
        ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80]); b = enc.tobytes()
        body += struct.pack(">I", len(b)) + b
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}],
                      "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    req = urllib.request.Request(URL, data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    r = urllib.request.urlopen(req, timeout=30)
    hl = int(r.headers["Inference-Header-Content-Length"]); raw = r.read()
    return np.frombuffer(raw[hl:], dtype="<f4")

res = collections.defaultdict(list)
for seed in range(300000, 300400):
    frames, lab = make_clip(seed)
    v = call(frames)
    code = int(v[0])
    # QC codes (dark/blurry/shaky) can legitimately mask a plate decision in the e2e path
    res[lab["kind"]].append((code, lab["present"], lab["done"], float(v[8]), float(v[9])))
tot = {"ok_no_plate": 0, "n_no_plate": 0, "ok_incomplete": 0, "n_incomplete": 0, "ok_good": 0, "n_good": 0}
for kind, rows in res.items():
    c = collections.Counter(PL.NAMES[r[0]] for r in rows)
    print(f"{kind:9s} n={len(rows):3d}  {dict(c)}")
    for code, p, d, pp, dd in rows:
        if p == 0: tot["n_no_plate"] += 1; tot["ok_no_plate"] += code == PL.OK
        elif d == 0: tot["n_incomplete"] += 1; tot["ok_incomplete"] += code == PL.OK
        else: tot["n_good"] += 1; tot["ok_good"] += code == PL.OK
print(f"\nsaid OK with NO plate in view      : {tot['ok_no_plate']}/{tot['n_no_plate']}")
print(f"said OK on incomplete/bad framing  : {tot['ok_incomplete']}/{tot['n_incomplete']}")
print(f"said OK on a genuinely good shot   : {tot['ok_good']}/{tot['n_good']}")
