"""Emulate 2x Jetson Orin Nano Super on the RTX PRO 4000: two worker processes, each capped to 8 SMs (1024 CUDA cores) by MPS
and the GPU clock locked at ~1 GHz. N simulated phones each send one request per second (skipping a tick while one is in flight,
as the app does). Latency = queue wait + inference, measured end to end. Preprocessing (JPEG decode) is NOT included.

usage: python bench_emul.py <model> [users...]     model: jepa16 | rtdetr_r18_320 | rtdetr_r18_640 | rtdetr_r50_640
env: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set for the workers here."""
import json
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np

MODEL = sys.argv[1]
USERS = [int(u) for u in sys.argv[2:]] or [1, 2, 4, 6, 8, 10, 12]
WORKERS = 2
DURATION = 20.0
RATE = int(os.environ.get("REQS_PER_TICK", "1"))      # >1 emulates the extra top/bottom checks (full cost each)


def build(model):
    import torch
    if model == "jepa16":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=torch.float16).cuda().eval()
        x = torch.randn(1, 16, 3, 256, 256, device="cuda", dtype=torch.float16)
        return lambda: m(pixel_values_videos=x).last_hidden_state
    from transformers import RTDetrForObjectDetection
    arch, res = model.split("_")[1], int(model.split("_")[2])
    m = RTDetrForObjectDetection.from_pretrained(f"PekingU/rtdetr_{arch}vd", torch_dtype=torch.float16).cuda().eval()
    x = torch.randn(1, 3, res, res, device="cuda", dtype=torch.float16)
    return lambda: m(pixel_values=x).logits


def worker(model, jobs, done, ready):
    import torch
    fn = build(model)
    with torch.no_grad():
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
        ready.put(os.getpid())
        while True:
            job = jobs.get()
            if job is None:
                return
            uid, t_sub = job
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            done.put((uid, t_sub, t0, time.time()))


def main():
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "11.5"          # 8 of 70 SMs = 1024 CUDA cores
    ctx = mp.get_context("spawn")
    jobs, done, ready = ctx.Queue(), ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(MODEL, jobs, done, ready)) for _ in range(WORKERS)]
    for p in procs:
        p.start()
    for _ in procs:
        ready.get(timeout=600)
    print(f"{MODEL}: {WORKERS} workers ready", flush=True)
    results = {}
    for n in USERS:
        in_flight = [0] * n
        lat, svc, completions = [], [], [[] for _ in range(n)]
        stop = time.time() + DURATION

        def phone(uid):
            t = time.time()
            while time.time() < stop:
                t += 1.0
                if in_flight[uid] == 0:
                    in_flight[uid] = RATE
                    for _ in range(RATE):
                        jobs.put((uid, time.time()))
                time.sleep(max(0.0, t - time.time()))

        threads = [threading.Thread(target=phone, args=(u,), daemon=True) for u in range(n)]
        for th in threads:
            th.start()
        end = stop + 5
        got = 0
        while time.time() < end:
            try:
                uid, t_sub, t0, t1 = done.get(timeout=0.5)
            except Exception:
                if time.time() > stop and all(v == 0 for v in in_flight):
                    break
                continue
            lat.append(t1 - t_sub)
            svc.append(t1 - t0)
            in_flight[uid] = max(0, in_flight[uid] - 1)
            if in_flight[uid] == 0:
                completions[uid].append(t1)
            got += 1
        for th in threads:
            th.join(timeout=1)
        while True:                                             # drain leftovers before the next point
            try:
                done.get(timeout=1.5)
            except Exception:
                break
        gaps = [g for c in completions for g in np.diff(c)] if any(len(c) > 1 for c in completions) else [float("nan")]
        r = dict(users=n, requests=got, lat_median_ms=1000 * float(np.median(lat)), lat_p95_ms=1000 * float(np.percentile(lat, 95)),
                 service_median_ms=1000 * float(np.median(svc)), refresh_median_s=float(np.nanmedian(gaps)), refresh_p95_s=float(np.nanpercentile(gaps, 95)))
        results[n] = r
        print(json.dumps(r), flush=True)
    for _ in procs:
        jobs.put(None)
    for p in procs:
        p.join(timeout=20)
    json.dump(results, open(f"emul_{MODEL}_x{RATE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
