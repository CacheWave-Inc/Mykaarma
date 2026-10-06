"""Train the plate head on cached frozen-JEPA features, then report honest held-out metrics."""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import plate_logic as PL

dev = "cuda"


def load(*splits):
    parts, labs = [], []
    for split in splits:
        f = np.load(f"plates/feats_{split}.npy", mmap_mode="r")
        for i in range(0, len(f), 1000):
            parts.append(torch.from_numpy(np.ascontiguousarray(f[i:i + 1000])).to(dev))
        labs.append(np.load(f"plates/labels_{split}.npz"))
    lab = {k: np.concatenate([l[k] for l in labs]) for k in labs[0].files}
    return torch.cat(parts), lab


def targets(lab):
    return (torch.tensor(lab["present"], dtype=torch.float32, device=dev),
            torch.tensor(lab["done"], dtype=torch.float32, device=dev),
            torch.tensor(lab["box"], dtype=torch.float32, device=dev))


def run_eval(head, X, T, bs=256):
    head.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(head(X[i:i + bs].float()))
    o = torch.cat(outs)
    return torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 1]), o[:, 2:6]


def iou(a, b):
    ix0, iy0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix1, iy1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def main():
    t0 = time.time()
    Xtr, ltr = load("train", "train2"); Xva, lva = load("val")
    Ptr, Dtr, Btr = targets(ltr); Pva, Dva, Bva = targets(lva)
    print(f"loaded {len(Xtr)} train / {len(Xva)} val in {time.time() - t0:.0f}s", flush=True)

    head = PL.Head().to(dev)
    epochs, bs = 30, 128
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=5e-2)
    steps = epochs * (len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    best, best_state = 1e9, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = Xtr[idx].float()
            x = x + 0.10 * x.std() * torch.randn_like(x)
            o = head(x)
            p, d, b = Ptr[idx], Dtr[idx], Btr[idx]
            loss = (F.binary_cross_entropy_with_logits(o[:, 0], p) + F.binary_cross_entropy_with_logits(o[:, 1], d)
                    + 5.0 * (F.smooth_l1_loss(o[:, 2:6], b, reduction="none").mean(1) * p).sum() / p.sum().clamp(min=1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        pp, dp, bp = run_eval(head, Xva, None)
        vloss = (F.binary_cross_entropy(pp, Pva) + F.binary_cross_entropy(dp, Dva)
                 + 5.0 * (F.smooth_l1_loss(bp, Bva, reduction="none").mean(1) * Pva).sum() / Pva.sum()).item()
        if vloss < best:
            best, best_state = vloss, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:2d}  train_loss {loss.item():.3f}  val_loss {vloss:.3f}  best {best:.3f}", flush=True)
    head.load_state_dict(best_state)
    torch.save(best_state, "plate_head.pt")
    del Xtr

    # ---- held-out test ----
    Xte, lte = load("test")
    pp, dp, bp = [t.cpu().numpy() for t in run_eval(head, Xte, None)]
    pres, done, box, kind = lte["present"], lte["done"], lte["box"], lte["kind"]
    rep = {}
    pred_p = pp >= PL.PRESENT_THR
    rep["present_accuracy"] = float((pred_p == (pres == 1)).mean())
    rep["present_recall"] = float(pred_p[pres == 1].mean())
    rep["present_false_alarm_on_no_plate"] = float(pred_p[pres == 0].mean())
    m = pres == 1
    rep["box_mean_iou_on_plates"] = float(iou(bp[m], box[m]).mean())

    codes = []
    for i in range(len(pp)):
        codes.append(PL.decide(float(pp[i]), float(dp[i]), tuple(bp[i]))[0])
    codes = np.array(codes)
    gt_codes = np.array([PL.decide(1.0 if pres[i] else 0.0, 1.0 if done[i] else 0.0, tuple(box[i]))[0] for i in range(len(pp))])
    said_ok = codes == PL.OK
    rep["says_OK_total"] = float(said_ok.mean())
    rep["FALSE_OK_when_no_plate"] = float(said_ok[pres == 0].mean())
    rep["FALSE_OK_when_plate_incomplete_or_badly_framed"] = float(said_ok[(pres == 1) & (done == 0)].mean())
    rep["OK_recall_on_good_shots"] = float(said_ok[done == 1].mean())
    rep["OK_precision"] = float((done[said_ok] == 1).mean()) if said_ok.any() else None
    directional = np.isin(gt_codes, [PL.LEFT, PL.RIGHT, PL.UP, PL.DOWN, PL.CLOSER, PL.BACK])
    rep["guidance_code_accuracy_on_plates_needing_a_move"] = float((codes[directional] == gt_codes[directional]).mean())
    rep["guidance_matches_by_kind"] = {k: float((codes[kind == k] == gt_codes[kind == k]).mean()) for k in ["framed", "partial", "far_near", "negative"]}
    rep["says_OK_by_kind"] = {k: float(said_ok[kind == k].mean()) for k in ["framed", "partial", "far_near", "negative"]}
    conf = {PL.NAMES[c]: int((codes == c).sum()) for c in np.unique(codes)}
    rep["predicted_code_counts"] = conf
    rep["n_test"] = int(len(pp))
    json.dump(rep, open("plate_eval.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
