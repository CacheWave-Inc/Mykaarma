"""Does the MPS thread percentage really limit the SMs? Measure fp16 matmul throughput and bandwidth under the current setting."""
import os
import time

import torch

pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(5):
    a @ b
torch.cuda.synchronize()
t0 = time.time(); n = 40
for _ in range(n):
    a @ b
torch.cuda.synchronize()
tflops = 2 * 4096 ** 3 * n / (time.time() - t0) / 1e12
x = torch.empty(512 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)
y = torch.empty_like(x)
for _ in range(3):
    y.copy_(x)
torch.cuda.synchronize()
t0 = time.time(); m = 20
for _ in range(m):
    y.copy_(x)
torch.cuda.synchronize()
gbs = 2 * x.numel() * 2 * m / (time.time() - t0) / 1e9
print(f"threads={pct:>5}%  fp16 matmul {tflops:6.1f} TFLOPS   memory copy {gbs:6.0f} GB/s")
