"""Load test for the RT-DETR edge server. A: N phones, each 1 request/s (like the app's score tick), 2x512 frames. B: closed-loop saturation sweep."""
import glob, io, json, random, struct, sys, threading, time, urllib.request, statistics
import numpy as np
from PIL import Image
BASE = sys.argv[1]; DUR = float(sys.argv[2]); MODE = sys.argv[3]            # MODE: realistic | sweep
TASK = int(sys.argv[4]) if len(sys.argv) > 4 else 3
random.seed(1)
imgs = glob.glob(__import__("os").environ.get("FRAMES_DIR", "frames") + "/**/*.jpg", recursive=True)
random.shuffle(imgs)
def enc(path):
    im = Image.open(path).convert("RGB").resize((512, 512)); b = io.BytesIO(); im.save(b, "JPEG", quality=85); return b.getvalue()
J = [enc(p) for p in imgs[:24]]
def payload():
    a, b = random.sample(J, 2)
    body = bytes([TASK, 2]) + b"".join(struct.pack(">I", len(j)) + j for j in (a, b))
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}],
                      "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    return hdr, body
def one():
    hdr, body = payload()
    req = urllib.request.Request(f"{BASE}/v2/models/rtdetr_inspect/infer", data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=30); r.read(); return (time.time() - t) * 1000, True
    except Exception:
        return (time.time() - t) * 1000, False
def stats(lat, errs, secs, label):
    lat = sorted(lat)
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")
    print(f"{label:34s} req={len(lat):5d} err={errs:3d} rate={len(lat) / secs:5.1f}/s  p50={pct(.5):6.0f} p95={pct(.95):6.0f} p99={pct(.99):6.0f} max={max(lat) if lat else 0:6.0f} ms", flush=True)
def run_users(n, secs, period):
    lat, errs, stop = [], [0], time.time() + secs
    def user(i):
        time.sleep(random.random() * period)
        nxt = time.time()
        while time.time() < stop:
            ms, ok = one(); lat.append(ms); errs[0] += (not ok)
            nxt += period; time.sleep(max(0.0, nxt - time.time()))        # one request per period; if late, goes again immediately (like the app: no overlap)
    ts = [threading.Thread(target=user, args=(i,)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return lat, errs[0]
def run_closed(n, secs):
    lat, errs, stop = [], [0], time.time() + secs
    def worker():
        while time.time() < stop:
            ms, ok = one(); lat.append(ms); errs[0] += (not ok)
    ts = [threading.Thread(target=worker) for _ in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return lat, errs[0]
for _ in range(4): one()
if MODE == "realistic":
    for n in (1, 5, 10, 15):
        lat, e = run_users(n, DUR, 1.0); stats(lat, e, DUR, f"{n:2d} phones x 1 req/s")
else:
    for n in (1, 2, 4, 10, 20):
        lat, e = run_closed(n, DUR); stats(lat, e, DUR, f"saturation, {n:2d} in flight")
