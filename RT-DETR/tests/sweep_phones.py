"""Latency vs number of phones (1..10), each phone = one scoring request per second like the app (no overlap).
usage: sweep_phones.py engine(jepa|rtdetr) mode(auto|single) secs_per_level out.json"""
import glob, io, json, random, struct, sys, threading, time, urllib.request
import numpy as np
from PIL import Image
ENGINE, MODE, SECS, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
BASE, MODEL, SIZE, NF = ("http://192.168.68.63:8000", "vjepa_inspect", 256, 16) if ENGINE == "jepa" else ("http://192.168.68.63:8100", "rtdetr_inspect", 512, 2)
random.seed(3)
imgs = glob.glob(__import__("os").environ.get("FRAMES_DIR", "frames") + "/**/*.jpg", recursive=True); random.shuffle(imgs)
def enc(p):
    b = io.BytesIO(); Image.open(p).convert("RGB").resize((SIZE, SIZE)).save(b, "JPEG", quality=85); return b.getvalue()
J = [enc(p) for p in imgs[:40]]
def one(task):
    fr = random.sample(J, NF)
    body = bytes([task, NF]) + b"".join(struct.pack(">I", len(j)) + j for j in fr)
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}], "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    r = urllib.request.Request(f"{BASE}/v2/models/{MODEL}/infer", data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    t = time.time()
    try:
        resp = urllib.request.urlopen(r, timeout=90); raw = resp.read(); ok = True
        srv = float(np.frombuffer(raw[int(resp.headers["Inference-Header-Content-Length"]):], dtype="<f4")[10])
    except Exception: ok, srv = False, float("nan")
    return (time.time() - t) * 1000, srv, ok
def level(n):
    rows, stop, warm = [], time.time() + SECS, time.time() + 3
    lock = threading.Lock()
    def phone(i):
        task = 3 if MODE == "auto" else [1, 2, 0][i % 3]
        time.sleep(random.random()); nxt = time.time()
        while time.time() < stop:
            ms, srv, ok = one(task)
            if time.time() > warm:
                with lock: rows.append((ms, srv, ok))
            nxt += 1.0; time.sleep(max(0.0, nxt - time.time()))
    ts = [threading.Thread(target=phone, args=(i,)) for i in range(n)]; [t.start() for t in ts]; [t.join() for t in ts]
    c = np.array([r[0] for r in rows]); s = np.array([r[1] for r in rows if r[1] == r[1]])
    return dict(n=n, req=len(rows), errs=sum(1 for r in rows if not r[2]), rps=len(rows) / (SECS - 3),
                p50=float(np.percentile(c, 50)), p95=float(np.percentile(c, 95)), mx=float(c.max()),
                srv50=float(np.percentile(s, 50)) if len(s) else None)
for _ in range(3): one(3)
res = []
for n in range(1, 11):
    r = level(n); res.append(r)
    print(f"{ENGINE} {MODE:6s} {n:2d} phones: {r['rps']:5.1f} req/s  round trip p50={r['p50']:6.0f} p95={r['p95']:6.0f} max={r['mx']:6.0f} ms  server p50={r['srv50'] or 0:5.0f}  errs={r['errs']}", flush=True)
    json.dump(res, open(OUT, "w"))
