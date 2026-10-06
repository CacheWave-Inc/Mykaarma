"""3-minute video through the V-JEPA 2 ViT-L encoder on one RTX PRO 4000: batching and model-split (pipeline) experiments.
VL-JEPA weights are gated, so the V-JEPA 2 encoder (its vision half) is used. Video is a synthetic 1080p 30 fps test clip.
"""
import os, time, threading, json
import cv2, numpy as np, torch
from transformers import AutoModel

VID = "test_3min_1080p.mp4"
FPS, SECS, W, H = 30, 180, 1920, 1080
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
out = {}

# ---------------------------------------------------------------- synthetic video
if not os.path.exists(VID):
    t0 = time.time()
    vw = cv2.VideoWriter(VID, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    base = np.zeros((H, W, 3), np.uint8)
    xs = np.linspace(0, 255, W).astype(np.uint8)
    base[:, :, 0] = xs[None, :]
    base[:, :, 1] = np.linspace(0, 255, H).astype(np.uint8)[:, None]
    rng = np.random.default_rng(0)
    for i in range(FPS * SECS):
        f = np.roll(base, i * 5, axis=1).copy()
        x = (i * 9) % (W - 300)
        cv2.rectangle(f, (x, 300), (x + 300, 700), (30, 200, 240), -1)
        f[::16, ::16] = rng.integers(0, 255, size=(len(range(0, H, 16)), len(range(0, W, 16)), 3), dtype=np.uint8)
        vw.write(f)
    vw.release()
    print(f"video written in {time.time()-t0:.0f}s, {os.path.getsize(VID)/1e6:.0f} MB")
out["video_mb"] = os.path.getsize(VID) / 1e6

# ---------------------------------------------------------------- decode + sample + resize on CPU
def prep(bgr):
    s = 256 / min(bgr.shape[:2])
    b = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    y, x = (b.shape[0] - 256) // 2, (b.shape[1] - 256) // 2
    rgb = cv2.cvtColor(b[y:y + 256, x:x + 256], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)

t0 = time.time()
cap = cv2.VideoCapture(VID)
step = FPS // 4                      # 30 fps video, 4 fps sampling -> every 7.5th; use 8 then 7 alternating via index math
keep = set(int(round(k * FPS / 4)) for k in range(SECS * 4))
frames, idx = [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if idx in keep:
        frames.append(prep(f))
    idx += 1
cap.release()
prep_s = time.time() - t0
clips = np.stack(frames)[: (len(frames) // 16) * 16].reshape(-1, 16, 3, 256, 256).astype(np.float16)
print(f"decode+sample+resize: {prep_s:.1f}s for {idx} frames -> {clips.shape[0]} clips")
out["prep_s"] = prep_s
out["n_clips"] = int(clips.shape[0])
X = torch.from_numpy(clips).cuda()

# ---------------------------------------------------------------- model
model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=torch.float16).cuda().eval()
enc = model.encoder
L = len(enc.layer)
half = L // 2


def stage1(x):
    h = enc.embeddings(x)
    for layer in enc.layer[:half]:
        h = layer(h, None)[0]
    return h


def stage2(h):
    for layer in enc.layer[half:]:
        h = layer(h, None)[0]
    return enc.layernorm(h)


def full(x):
    return stage2(stage1(x))


@torch.no_grad()
def timed(fn, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - t


N = X.shape[0]
with torch.no_grad():
    # sanity: split == full model encoder output
    a = full(X[:1]); b = model(X[:1], skip_predictor=True).last_hidden_state
    out["split_matches_full_maxdiff"] = float((a - b).abs().max())

    # A) sequential, batch 1
    def run_bs(bs):
        for i in range(0, N, bs):
            full(X[i:i + bs])
    res = {}
    for bs in (1, 2, 4, 8, 15):
        res[bs] = timed(lambda bs=bs: run_bs(bs))
        print(f"batch {bs:2d}: {res[bs]:.2f}s for {N} clips -> {res[bs]/N*1000:.1f} ms/clip")
    out["batch_s"] = {str(k): v for k, v in res.items()}

    # stage times
    t1 = timed(lambda: [stage1(X[i:i + 1]) for i in range(N)]) / N * 1000
    h1 = stage1(X[:1])
    t2 = timed(lambda: [stage2(h1) for _ in range(N)]) / N * 1000
    out["stage1_ms"], out["stage2_ms"] = t1, t2
    print(f"stage 1 (embed + layers 0-{half-1}) {t1:.1f} ms, stage 2 (layers {half}-{L-1}) {t2:.1f} ms per clip")

    # B) model split, two CUDA streams on the SAME GPU, pipelined by two threads
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    q, done, lock = [], [], threading.Lock()
    import queue
    qu = queue.Queue()

    def worker1():
        with torch.cuda.stream(s1):
            for i in range(N):
                h = stage1(X[i:i + 1])
                ev = torch.cuda.Event(); ev.record(s1)
                qu.put((h, ev))
        qu.put(None)

    def worker2():
        with torch.cuda.stream(s2):
            while True:
                item = qu.get()
                if item is None:
                    break
                h, ev = item
                s2.wait_event(ev)
                stage2(h)

    def pipe():
        th = [threading.Thread(target=worker1), threading.Thread(target=worker2)]
        [t.start() for t in th]; [t.join() for t in th]
    pipe(); torch.cuda.synchronize()
    t = time.time(); pipe(); torch.cuda.synchronize()
    out["pipeline_1gpu_s"] = time.time() - t
    print(f"model split, 2 stages pipelined on ONE GPU: {out['pipeline_1gpu_s']:.2f}s")

out["N"] = N
out["two_gpu_projection_s"] = (N + 1) * max(t1, t2) / 1000
out["one_gpu_seq_s"] = res[1]
print(json.dumps(out, indent=1))
json.dump(out, open("split_results.json", "w"), indent=1)
