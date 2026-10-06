"""Emulate N phones each running a full inspection: battery, plate and brake in a random order, 12-20 s on each item at 1 scoring request/s
(2x512 frames), then one 'verify' request (bit 7, 1 frame) for the captured still, then the next item; when all three are done the phone starts again.
Prints the server-side latency stats every 30 s. usage: emu_inspect.py base n_phones secs"""
import glob, io, json, random, struct, sys, threading, time, urllib.request
import numpy as np
from PIL import Image
BASE, N, SECS = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
random.seed()
imgs = glob.glob(__import__("os").environ.get("FRAMES_DIR", "frames") + "/**/*.jpg", recursive=True); random.shuffle(imgs)
def enc(p):
    b = io.BytesIO(); Image.open(p).convert("RGB").resize((512, 512)).save(b, "JPEG", quality=85); return b.getvalue()
J = [enc(p) for p in imgs[:30]]
NAMES = {0: "brake", 1: "battery", 2: "plate"}
lat, errs, lock = [], [0], threading.Lock()
def req(pid, task, nframes, verify=False):
    fr = random.sample(J, nframes)
    body = bytes([task | (0x80 if verify else 0), nframes]) + b"".join(struct.pack(">I", len(j)) + j for j in fr)
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}], "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    r = urllib.request.Request(f"{BASE}/v2/models/rtdetr_inspect/infer", data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    t = time.time()
    srv = float("nan")
    try:
        resp = urllib.request.urlopen(r, timeout=30); raw = resp.read(); ok = True
        srv = float(np.frombuffer(raw[int(resp.headers["Inference-Header-Content-Length"]):], dtype="<f4")[10])      # server-reported handling time
    except Exception: ok = False
    with lock:
        lat.append((time.time(), (time.time() - t) * 1000, srv, pid, task, verify)); errs[0] += (not ok)
stop = time.time() + SECS
def phone(i):
    time.sleep(random.random() * 5)
    while time.time() < stop:
        order = [0, 1, 2]; random.shuffle(order)
        for task in order:
            end = time.time() + random.uniform(12, 20); nxt = time.time()
            while time.time() < min(end, stop):
                req(i, task, 2); nxt += 1.0; time.sleep(max(0.0, nxt - time.time()))
            if time.time() < stop: req(i, task, 1, verify=True)          # the accepted still is re-checked once
            time.sleep(1.5)                                           # technician moves to the next item
ts = [threading.Thread(target=phone, args=(i,), daemon=True) for i in range(N)]; [t.start() for t in ts]
t0 = time.time(); print(f"{N} emulated inspecting phones started", flush=True)
q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
while time.time() < stop:
    time.sleep(30)
    with lock:
        cut = time.time() - 30; rows = [r for r in lat if r[0] >= cut]; e = errs[0]
    if not rows: continue
    c = [r[1] for r in rows]; sv = [r[2] for r in rows if r[2] == r[2]]
    print(f"[{int(time.time() - t0):4d}s] {len(rows)/30:4.1f} req/s | phone-side round trip p50={q(c,50):4.0f} p95={q(c,95):4.0f} max={max(c):4.0f} | server-reported p50={q(sv,50):4.0f} p95={q(sv,95):4.0f} max={max(sv):4.0f} ms | errors={e}", flush=True)
    byp = " ".join(f"P{p}:{q([r[2] for r in rows if r[3]==p and r[2]==r[2]],50):.0f}" for p in range(N))
    byt = " ".join(f"{NAMES[t]}:{q([r[2] for r in rows if r[4]==t and not r[5] and r[2]==r[2]],50):.0f}" for t in (1,2,0))
    ver = [r[2] for r in rows if r[5] and r[2]==r[2]]
    print("        server ms by phone (median): " + byp, flush=True)
    print("        by item: " + byt + f" verify:{q(ver,50):.0f}", flush=True)
