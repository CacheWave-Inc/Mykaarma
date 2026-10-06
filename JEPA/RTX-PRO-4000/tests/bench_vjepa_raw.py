"""Raw PyTorch (no Triton) latency benchmark: V-JEPA2 ViT-L forward pass on the GPU.
Same input shape as the earlier Triton benchmark: 1 x 16 x 3 x 256 x 256, fp16.
Reports GPU-only compute time (model.forward, properly synchronized), not
network/HTTP overhead, since Triton isn't currently deployed on this freshly
reimaged box.
"""
import time
import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEVICE = "cuda"

print(f"Loading {MODEL_ID} ...")
t0 = time.time()
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
print(f"Loaded in {time.time()-t0:.1f}s")
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e6:.1f}M")

x = torch.from_numpy(np.random.randn(1, 16, 3, 256, 256).astype(np.float16)).to(DEVICE)

# warmup
with torch.no_grad():
    for _ in range(5):
        _ = model(pixel_values_videos=x)
    torch.cuda.synchronize()

N = 50
lat = []
with torch.no_grad():
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(pixel_values_videos=x)
        torch.cuda.synchronize()
        lat.append((time.time() - t0) * 1000)

lat = np.array(lat)
print(f"\nBatch=1, N={N} runs")
print(f"  mean {lat.mean():.1f} ms  p50 {np.percentile(lat,50):.1f} ms  p99 {np.percentile(lat,99):.1f} ms  min {lat.min():.1f} ms")
print(f"  throughput at saturation (1/mean): {1000/lat.mean():.2f} clips/s (single stream, not batched)")

# batch=8 to see batched throughput (Triton's preferred batch size was 4/8)
xb = torch.from_numpy(np.random.randn(8, 16, 3, 256, 256).astype(np.float16)).to(DEVICE)
with torch.no_grad():
    for _ in range(3):
        _ = model(pixel_values_videos=xb)
    torch.cuda.synchronize()
    lat8 = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(pixel_values_videos=xb)
        torch.cuda.synchronize()
        lat8.append((time.time() - t0) * 1000)
lat8 = np.array(lat8)
print(f"\nBatch=8, N=20 runs")
print(f"  mean {lat8.mean():.1f} ms per batch  ->  {lat8.mean()/8:.2f} ms/clip  ->  {8000/lat8.mean():.2f} clips/s")
print(f"\nGPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
