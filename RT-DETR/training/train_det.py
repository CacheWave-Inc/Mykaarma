"""train_det.py <task> [epochs] [max_train] : fine-tune RT-DETR-R18 (COCO-pretrained) on the synthetic single-frame data.
task: plate | brake | battery  (battery has 2 classes: bare battery, covered battery). Writes rtdetr_<task>/ + rtdetr_<task>_eval.json.
Decision rules on the detected box mirror obj_logic / plate_logic so the two engines are directly comparable."""
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RTDetrForObjectDetection

import det_logic as DL

TASK = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MAX_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
import os
SUF = os.environ.get("OUTSUF", "")
MODE = sys.argv[4] if len(sys.argv) > 4 else ""        # battery only: "bare" (bare battery) or "covered" (battery with its cover on)
CLASSES = ["battery_covered"] if MODE == "covered" else ["battery", "battery_covered"] if (TASK == "battery" and MODE == "") else [TASK]
OUT = ("rtdetr_battery_covered" if MODE == "covered" else f"rtdetr_{TASK}") + SUF
import os
R = int(os.environ.get("RES", "320"))
LRB = float(os.environ.get("LRB", "1e-5"))
CKPT = os.environ.get("CKPT", "PekingU/rtdetr_r18vd")
AUGLO, AUGHI = [float(v) for v in os.environ.get("AUGB", "0.6,1.4").split(",")]
SUF = os.environ.get("OUTSUF", "")
BS = 32
dev = "cuda"


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def to_label(box, cls):
    x0, y0, x1, y1 = [min(max(float(v), 0.0), 1.0) for v in box]
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        return None
    return cls, ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def load(split):
    d = pickle.load(open(f"local/{TASK}_{split}.pkl", "rb"))
    n = min(len(d["jpg"]), MAX_TRAIN) if split == "train" else len(d["jpg"])
    with Pool(15) as p:
        imgs = np.stack(p.map(dec, d["jpg"][:n], chunksize=64))
    labels = []
    for i in range(n):
        items = []
        if MODE != "covered" and d["present"][i]:
            lab = to_label(d["box"][i], 0)
            if lab:
                items.append(lab)
        if TASK == "battery" and d["covered"][i] and MODE != "bare":
            lab = to_label(d["cbox"][i], 0 if MODE == "covered" else 1)
            if lab:
                items.append(lab)
        labels.append(items)
    return torch.from_numpy(imgs), labels, d, n


def batch_labels(items_list, flip=None):
    out = []
    for k, items in enumerate(items_list):
        cl = torch.tensor([c for c, _ in items], dtype=torch.long, device=dev)
        bx = torch.tensor([b for _, b in items], dtype=torch.float32, device=dev).reshape(-1, 4)
        if flip is not None and flip[k] and len(items):
            bx[:, 0] = 1 - bx[:, 0]
        out.append({"class_labels": cl, "boxes": bx})
    return out


def prep(x_uint8, aug):
    x = x_uint8.to(dev).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False)
    if aug:
        b = x.shape[0]
        x = x * (AUGLO + (AUGHI - AUGLO) * torch.rand(b, 1, 1, 1, device=dev)) + 0.15 * (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
        x = x * (0.9 + 0.2 * torch.rand(b, 3, 1, 1, device=dev))
        g = x.mean(1, keepdim=True)
        x = g + (x - g) * (0.6 + 0.8 * torch.rand(b, 1, 1, 1, device=dev))
        blur = torch.rand(b, device=dev) < 0.25
        if blur.any():
            x[blur] = F.conv2d(x[blur], torch.ones(3, 1, 5, 5, device=dev) / 25, padding=2, groups=3)
        x = x + torch.randn_like(x) * 0.02 * torch.rand(b, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


def load_real():
    """(frames uint8 [n,512,512,3] on the GPU, labels) from REAL_DIR / REAL_JSON, or (None, None)."""
    import json as _json
    if not os.environ.get("REAL_JSON"):
        return None, None
    lab = _json.load(open(os.environ["REAL_JSON"]))
    imgs, labels = [], []
    for name, v in sorted(lab.items()):
        sc = v.get("battery", {}).get("score", 0.0)
        if 0.12 <= sc < 0.20:
            continue
        im = cv2.cvtColor(cv2.imread(os.path.join(os.environ["REAL_DIR"], name)), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        items = []
        if sc >= 0.20:
            x0, y0, x1, y1 = v["battery"]["box"]
            got = to_label((x0 / w, y0 / h, x1 / w, y1 / h), 0)
            if got:
                items.append(got)
        imgs.append(im)
        labels.append(items)
    pos = sum(1 for l in labels if l)
    print(f"real frames: {len(imgs)} ({pos} with a battery box, {len(imgs) - pos} negatives)", flush=True)
    return torch.from_numpy(np.stack(imgs)).to(dev), labels


def main():
    t0 = time.time()
    Xtr, Ltr, _, ntr = load("train")
    Xva, Lva, _, _ = load("val")
    print(f"{TASK}: {ntr} train / {len(Xva)} val loaded in {time.time() - t0:.0f}s, classes={CLASSES}", flush=True)
    model = RTDetrForObjectDetection.from_pretrained(CKPT, num_labels=len(CLASSES), ignore_mismatched_sizes=True).to(dev)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": LRB}, {"params": rest, "lr": 1e-4}], weight_decay=1e-4)
    steps = EPOCHS * (ntr // BS)
    base = [LRB, 1e-4]

    def set_lr(it):
        f = min(1.0, (it + 1) / 200) * 0.5 * (1 + math.cos(math.pi * it / steps))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(f, 0.02)

    Xtr_g = Xtr.to(dev)
    Xreal, Lreal = load_real()
    best, best_state, it = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(ntr).tolist()
        for i in range(0, ntr - BS + 1, BS):
            idx = perm[i:i + BS]
            if Xreal is not None:                       # 1 in 4 samples of every batch is a real frame
                ridx = np.random.randint(0, len(Lreal), BS // 4).tolist()
                idx = idx[:BS - len(ridx)]
            flip = [(TASK != "plate" and np.random.rand() < 0.5) for _ in range(BS)]
            x = prep(Xtr_g[idx], True)
            items = [Ltr[j] for j in idx]
            if Xreal is not None:
                x = torch.cat([x, prep(Xreal[ridx], True)])
                items += [Lreal[j] for j in ridx]
            fl = torch.tensor(flip[:len(x)], device=dev)
            x = torch.where(fl.view(-1, 1, 1, 1), x.flip(3), x)
            labels = batch_labels(items, flip)
            set_lr(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=x, labels=labels)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); it += 1
            if it % 100 == 0:
                print(f"  it {it}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        vl = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(Xva) - BS + 1, BS):
                vl.append(model(pixel_values=prep(Xva[i:i + BS], False), labels=batch_labels(Lva[i:i + BS])).loss.item())
        v = float(np.mean(vl))
        if v < best:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"epoch {ep} train_loss {loss.item():.3f} val_loss {v:.3f} best {best:.3f} {time.time() - t0:.0f}s", flush=True)
    if os.environ.get("LAST") != "1":      # LAST=1: keep the final (fully trained) weights; val loss is a poor guide for the battery task
        model.load_state_dict(best_state)
    model.save_pretrained(OUT)
    json.dump({"res": R}, open(os.path.join(OUT, "cw_res.json"), "w"))
    del Xtr_g

    # ---- held-out test with the same decision rules ----
    Xte, Lte, d, nte = load("test")
    scores = np.zeros((nte, len(CLASSES))); boxes = np.zeros((nte, len(CLASSES), 4))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, nte, 64):
            out = model(pixel_values=prep(Xte[i:i + 64], False))
            sc = out.logits.float().sigmoid()
            for c in range(len(CLASSES)):
                s, q = sc[:, :, c].max(1)
                b = out.pred_boxes.float()[torch.arange(len(q)), q]
                scores[i:i + 64, c] = s.cpu().numpy()
                cx, cy, w, h = b.unbind(1)
                boxes[i:i + 64, c] = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).cpu().numpy()
    pres, done, kind = d["present"][:nte], d["done"][:nte], d["kind"][:nte]
    cov = d["covered"][:nte]
    gtbox = np.clip(d["box"][:nte], 0, 1)
    rep = {"classes": CLASSES, "n_test": int(nte)}
    if MODE == "covered":
        cb = np.clip(d["cbox"][:nte], 0, 1)
        sw = {}
        for thr in (0.3, 0.5, 0.7, 0.85):
            hit = scores[:, 0] >= thr
            sw[str(thr)] = dict(covered_recall=float(hit[cov == 1].mean()), false_alarm_on_not_covered=float(hit[cov == 0].mean()),
                                false_alarm_on_bare_battery=float(hit[pres == 1].mean()))
        ious = []
        for i in np.where(cov == 1)[0]:
            a, b = boxes[i, 0], cb[i]
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            ious.append(inter / max(ua, 1e-9))
        rep["box_mean_iou"] = float(np.mean(ious)); rep["threshold_sweep"] = sw
        json.dump(rep, open(f"{OUT}_eval.json", "w"), indent=1)
        print(json.dumps(rep, indent=1)); print(f"total {time.time() - t0:.0f}s")
        return
    sweep = {}
    for thr in (0.3, 0.5, 0.7, 0.85):
        codes = np.array([DL.decide(TASK, scores[i, 0], boxes[i, 0], scores[i, 1] if (TASK == "battery" and len(CLASSES) > 1) else 0.0, thr) for i in range(nte)])
        gt = np.array([DL.decide(TASK, 1.0 if pres[i] and done[i] else (0.9 if pres[i] else 0.0), gtbox[i], 0.0, 0.5, gt_mode=True) for i in range(nte)])
        ok = codes == DL.OK
        r = dict(ok_on_good=float(ok[done == 1].mean()), false_ok_bad=float(ok[(pres == 1) & (done == 0)].mean()),
                 false_ok_none=float(ok[pres == 0].mean()),
                 present_acc=float(((scores[:, 0] >= thr) == (pres == 1)).mean()))
        mv = np.isin(gt, [DL.LEFT, DL.RIGHT, DL.UP, DL.DOWN, DL.CLOSER, DL.BACK])
        r["guidance_acc_when_move_needed"] = float((codes[mv] == gt[mv]).mean()) if mv.any() else None
        if TASK == "battery" and len(CLASSES) > 1:
            r["covered_flagged"] = float((codes[cov == 1] == DL.COVERED).mean())
            r["bare_flagged_covered"] = float((codes[pres == 1] == DL.COVERED).mean())
        sweep[str(thr)] = r
    m = pres == 1
    ious = []
    for i in np.where(m)[0]:
        a, b = boxes[i, 0], gtbox[i]
        inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        ious.append(inter / max(ua, 1e-9))
    rep["box_mean_iou"] = float(np.mean(ious))
    rep["threshold_sweep"] = sweep
    json.dump(rep, open(f"{OUT}_eval.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
