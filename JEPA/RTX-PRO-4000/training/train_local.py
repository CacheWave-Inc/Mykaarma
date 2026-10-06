"""train_local.py : one small MobileNetV3 model for brake / battery / plate (single frame), then ONNX export + held-out report.

Outputs (19 logits): task t in 0 brake, 1 battery, 2 plate uses [6t .. 6t+5] = present, done, x0, y0, x1, y1 ; [18] = battery 'covered'.
Input: float32 [1,3,224,224] RGB scaled 0..1 (normalisation lives inside the graph)."""
import json
import pickle
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import obj_logic as OL
import plate_logic as PL

R = 224
TASKS = ["brake", "battery", "plate"]
dev = "cuda"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def dec(b):
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    return cv2.resize(im, (R, R), interpolation=cv2.INTER_AREA)


def load(split):
    pool = Pool(15)
    X, P, D, B, C, T, K = [], [], [], [], [], [], []
    for t, name in enumerate(TASKS):
        d = pickle.load(open(f"local/{name}_{split}.pkl", "rb"))
        X.append(np.stack(pool.map(dec, d["jpg"], chunksize=64)))
        P.append(d["present"]); D.append(d["done"]); B.append(d["box"]); C.append(d["covered"]); K.append(d["kind"])
        T.append(np.full(len(d["present"]), t, np.int64))
    pool.close()
    cat = np.concatenate
    return (torch.from_numpy(cat(X)), torch.from_numpy(cat(P)).float(), torch.from_numpy(cat(D)).float(),
            torch.from_numpy(cat(B)).float(), torch.from_numpy(cat(C)).float(), torch.from_numpy(cat(T)), cat(K))


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.mobilenet_v3_small(weights=w).features       # [B,576,7,7]
        self.reduce = nn.Sequential(nn.Conv2d(576, 96, 1), nn.BatchNorm2d(96), nn.Hardswish())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(96 * 49, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 19))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                      # x: [B,3,224,224] float 0..1
        return self.fc(self.reduce(self.backbone((x - self.mean) / self.std)))


def pick(o, t):
    idx = (t * 6).unsqueeze(1) + torch.arange(6, device=o.device).unsqueeze(0)
    return o.gather(1, idx)


def augment(x):
    b = x.shape[0]
    x = x * (0.6 + 0.8 * torch.rand(b, 1, 1, 1, device=dev)) + 0.15 * (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
    x = x * (0.9 + 0.2 * torch.rand(b, 3, 1, 1, device=dev))
    gray = x.mean(1, keepdim=True)
    s = 0.6 + 0.8 * torch.rand(b, 1, 1, 1, device=dev)
    x = gray + (x - gray) * s
    blur = torch.rand(b, device=dev) < 0.25
    if blur.any():
        k = torch.ones(3, 1, 5, 5, device=dev) / 25
        x[blur] = F.conv2d(x[blur], k, padding=2, groups=3)
    x = x + torch.randn_like(x) * 0.02 * torch.rand(b, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


def lossf(o, t, p, d, b, c):
    s = pick(o, t)
    l = F.binary_cross_entropy_with_logits(s[:, 0], p) + F.binary_cross_entropy_with_logits(s[:, 1], d)
    l = l + 5.0 * (F.smooth_l1_loss(s[:, 2:6], b, reduction="none").mean(1) * p).sum() / p.sum().clamp(min=1)
    bat = t == 1
    if bat.any():
        l = l + F.binary_cross_entropy_with_logits(o[bat, 18], c[bat])
    return l


def predict(net, X, bs=256):
    net.eval()
    outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev).permute(0, 3, 1, 2).float() / 255
            outs.append(net(xb).float().cpu())
    return torch.cat(outs)


def main():
    t0 = time.time()
    Xtr, Ptr, Dtr, Btr, Ctr, Ttr, _ = load("train")
    Xva, Pva, Dva, Bva, Cva, Tva, _ = load("val")
    print(f"decoded {len(Xtr)} train / {len(Xva)} val in {time.time() - t0:.0f}s", flush=True)
    try:
        net = Net(True).to(dev)
        print("pretrained ImageNet backbone", flush=True)
    except Exception as e:
        print("pretrained weights unavailable, training from scratch:", repr(e)[:120], flush=True)
        net = Net(False).to(dev)
    Xtr_g = Xtr.to(dev)
    Ptr, Dtr, Btr, Ctr, Ttr = [v.to(dev) for v in (Ptr, Dtr, Btr, Ctr, Ttr)]
    bs = 128
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=EPOCHS * (len(Xtr) // bs), pct_start=0.15)
    best, best_state = 1e9, None
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = Xtr_g[idx].permute(0, 3, 1, 2).float() / 255
            x = augment(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x)
            loss = lossf(o.float(), Ttr[idx], Ptr[idx], Dtr[idx], Btr[idx], Ctr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ov = predict(net, Xva)
        vloss = lossf(ov, Tva, Pva, Dva, Bva, Cva).item()
        if vloss < best:
            best, best_state = vloss, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep:2d} train_loss {loss.item():.3f} val_loss {vloss:.3f} best {best:.3f}  {time.time() - t0:.0f}s", flush=True)
    net.load_state_dict(best_state)
    torch.save(best_state, "local_model.pt")
    del Xtr_g

    # ---- held-out test, same metrics as the server model ----
    Xte, Pte, Dte, Bte, Cte, Tte, Kte = load("test")
    o = predict(net, Xte)
    sig = lambda v: 1 / (1 + torch.exp(-v))
    rep = {}
    for t, name in enumerate(TASKS):
        m = (Tte == t).numpy()
        s = pick(o, Tte)[m]
        pp, dp, bp = sig(s[:, 0]).numpy(), sig(s[:, 1]).numpy(), s[:, 2:6].numpy()
        cp = sig(o[m, 18]).numpy() if name == "battery" else np.zeros(m.sum())
        pres, done, box, kind = Pte[m].numpy(), Dte[m].numpy(), Bte[m].numpy(), Kte[m]
        covl = Cte[m].numpy()
        r = {}
        r["present_accuracy"] = float(((pp >= 0.85) == (pres == 1)).mean())
        r["present_false_alarm_on_no_target"] = float((pp[pres == 0] >= 0.85).mean())
        inter = np.clip(np.minimum(bp[:, 2], box[:, 2]) - np.maximum(bp[:, 0], box[:, 0]), 0, None) * np.clip(np.minimum(bp[:, 3], box[:, 3]) - np.maximum(bp[:, 1], box[:, 1]), 0, None)
        ua = (bp[:, 2] - bp[:, 0]) * (bp[:, 3] - bp[:, 1]) + (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) - inter
        r["box_mean_iou"] = float((inter / np.maximum(ua, 1e-9))[pres == 1].mean())
        sweep = {}
        for thr in (0.95, 0.9, 0.8, 0.6, 0.4):
            if name == "plate":
                codes = np.array([PL.decide(float(pp[i]), float(dp[i]), tuple(bp[i]), done_thr=thr)[0] for i in range(len(pp))])
            else:
                codes = np.array([OL.decide(name, float(pp[i]), float(dp[i]), tuple(bp[i]), float(cp[i]), done_thr=thr)[0] for i in range(len(pp))])
            ok = codes == OL.OK
            sweep[str(thr)] = dict(ok_on_good=float(ok[done == 1].mean()), false_ok_bad=float(ok[(pres == 1) & (done == 0)].mean()),
                                   false_ok_none=float(ok[pres == 0].mean()))
        r["done_threshold_sweep"] = sweep
        if name == "battery":
            codes = np.array([OL.decide(name, float(pp[i]), float(dp[i]), tuple(bp[i]), float(cp[i]), done_thr=0.6)[0] for i in range(len(pp))])
            r["covered_flagged"] = float((codes[covl == 1] == OL.COVERED).mean())
            r["bare_flagged_covered"] = float((codes[pres == 1] == OL.COVERED).mean())
        rep[name] = r
    json.dump(rep, open("local_eval.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))

    # ---- ONNX export ----
    net.eval().cpu()
    dummy = torch.rand(1, 3, R, R)
    torch.onnx.export(net, dummy, "inspector_local.onnx", input_names=["image"], output_names=["out"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession("inspector_local.onnx", providers=["CPUExecutionProvider"])
    x = torch.rand(4, 3, R, R)
    a = net(x[:1]).detach().numpy()
    b = sess.run(None, {"image": x[:1].numpy()})[0]
    print("onnx max abs diff vs torch:", float(np.abs(a - b).max()), "size MB:", round(__import__("os").path.getsize("inspector_local.onnx") / 1e6, 2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
