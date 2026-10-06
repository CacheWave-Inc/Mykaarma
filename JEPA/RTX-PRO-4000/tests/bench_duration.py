"""Real single-pass latency vs. clip duration, 10s steps up to 180s, on the actual
RTX PRO 4000 GPU. Same 4 fps / 256x256 sampling convention as slides 3-4 (fixed
sample rate; longer clip duration = more frames fed in one forward pass, model
accepts variable frame counts as verified). No interpolation -- every point is a
real timed GPU run, median of several reps, written incrementally so partial
results survive an OOM at the high end.
"""
import time, json, gc
import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEVICE = "cuda"
FPS_SAMPLE = 4  # frames per second sampled, per slide 3's convention

print(f"Loading {MODEL_ID} ...")
t0 = time.time()
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
print(f"Loaded in {time.time()-t0:.1f}s")

durations = list(range(10, 181, 10))  # 10,20,...,180
results = []

for dur_s in durations:
    n_frames = dur_s * FPS_SAMPLE
    torch.cuda.empty_cache()
    gc.collect()
    try:
        x = torch.from_numpy(np.random.randn(1, n_frames, 3, 256, 256).astype(np.float16)).to(DEVICE)
        reps = 5 if dur_s <= 60 else (3 if dur_s <= 120 else 2)
        with torch.no_grad():
            # warmup
            for _ in range(1):
                _ = model(pixel_values_videos=x)
            torch.cuda.synchronize()
            lat = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.time()
                out = model(pixel_values_videos=x)
                torch.cuda.synchronize()
                lat.append((time.time() - t0) * 1000)
        lat = np.array(lat)
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        row = dict(duration_s=dur_s, n_frames=n_frames, mean_ms=float(lat.mean()),
                   median_ms=float(np.median(lat)), min_ms=float(lat.min()),
                   max_ms=float(lat.max()), reps=reps, peak_mem_gb=round(mem_gb, 2),
                   status="ok")
        print(f"dur={dur_s:>3}s  frames={n_frames:>3}  median={row['median_ms']:>8.1f} ms  "
              f"mean={row['mean_ms']:>8.1f} ms  peak_mem={mem_gb:.2f} GB")
        del x, out
    except RuntimeError as e:
        row = dict(duration_s=dur_s, n_frames=n_frames, status="failed", error=str(e)[:300])
        print(f"dur={dur_s:>3}s  frames={n_frames:>3}  FAILED: {str(e)[:200]}")
        torch.cuda.empty_cache()
    results.append(row)
    json.dump(results, open("bench_duration_results.json", "w"), indent=1)
    torch.cuda.reset_peak_memory_stats()

print("\nDone. Results in bench_duration_results.json")
print(json.dumps(results, indent=1))
