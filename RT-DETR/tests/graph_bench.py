import json, os, sys, time, torch
from transformers import RTDetrForObjectDetection
for name in ("plate", "brake", "battery"):
    src = os.path.expanduser(f"~/rtdetr/models/rtdetr_{name}")
    rf = os.path.join(src, "cw_res.json"); res = json.load(open(rf))["res"] if os.path.exists(rf) else 320
    m = RTDetrForObjectDetection.from_pretrained(src, dtype=torch.float16).cuda().eval()
    x = torch.rand(1, 3, res, res, device="cuda", dtype=torch.float16)
    def eager(inp):
        o = m(pixel_values=inp); return o.logits, o.pred_boxes
    with torch.no_grad():
        for _ in range(5): eager(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(30): ref = eager(x)
        torch.cuda.synchronize(); te = (time.time() - t) / 30 * 1000
        static = x.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): eager(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = eager(static)
        xs = torch.rand(1, 3, res, res, device="cuda", dtype=torch.float16)
        static.copy_(xs); g.replay(); torch.cuda.synchronize()
        want = eager(xs)
        diff = max((out[0].float() - want[0].float()).abs().max().item(), (out[1].float() - want[1].float()).abs().max().item())
        t = time.time()
        for _ in range(30):
            static.copy_(xs); g.replay()
        torch.cuda.synchronize(); tg = (time.time() - t) / 30 * 1000
    print(f"{name:8s} res={res}  eager {te:6.1f} ms   cuda-graph {tg:6.1f} ms   max|diff| {diff:.5f}", flush=True)
    del m, g
