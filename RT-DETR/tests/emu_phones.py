"""Emulate extra phones: one thread per entry in TASKS, each 1 request/s (2x512 frames, like the app). usage: emu_phones.py base secs tasks(e.g. 1,2,0,1 or 3,3,3,3)"""
import glob, io, json, random, struct, sys, threading, time, urllib.request
from PIL import Image
BASE, SECS, TASKS = sys.argv[1], float(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
random.seed(2)
imgs = glob.glob(__import__("os").environ.get("FRAMES_DIR", "frames") + "/**/*.jpg", recursive=True); random.shuffle(imgs)
def enc(p):
    b = io.BytesIO(); Image.open(p).convert("RGB").resize((512, 512)).save(b, "JPEG", quality=85); return b.getvalue()
J = [enc(p) for p in imgs[:24]]
def one(task):
    a, b = random.sample(J, 2)
    body = bytes([task, 2]) + b"".join(struct.pack(">I", len(j)) + j for j in (a, b))
    hdr = json.dumps({"inputs": [{"name": "FRAMES", "shape": [len(body)], "datatype": "UINT8", "parameters": {"binary_data_size": len(body)}}], "outputs": [{"name": "GUIDANCE", "parameters": {"binary_data": True}}]}).encode()
    req = urllib.request.Request(f"{BASE}/v2/models/rtdetr_inspect/infer", data=hdr + body, headers={"Inference-Header-Content-Length": str(len(hdr))})
    t = time.time()
    try: urllib.request.urlopen(req, timeout=30).read(); return (time.time() - t) * 1000, True
    except Exception: return (time.time() - t) * 1000, False
lat, errs, stop = [], [0], time.time() + SECS
def phone(task):
    time.sleep(random.random()); nxt = time.time()
    while time.time() < stop:
        ms, ok = one(task); lat.append(ms); errs[0] += (not ok)
        nxt += 1.0; time.sleep(max(0.0, nxt - time.time()))
ts = [threading.Thread(target=phone, args=(t,)) for t in TASKS]; [t.start() for t in ts]; [t.join() for t in ts]
lat.sort(); p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
print(f"emulated {len(TASKS)} phones (tasks {TASKS}): req={len(lat)} err={errs[0]} p50={p(.5):.0f} p95={p(.95):.0f} max={lat[-1]:.0f} ms")
