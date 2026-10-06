"""train_obj.py <task>: train the head on cached features, write <task>_head.pt + <task>_eval.json (held-out test)."""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import obj_logic as OL

TASK = sys.argv[1]
dev = "cuda"


def load(split):
    f = np.load(f"plates/feats_{TASK}_{split}.npy", mmap_mode="r")
    X = torch.cat([torch.from_numpy(np.ascontiguousarray(f[i:i + 1000])).to(dev) for i in range(0, len(f), 1000)])
    return X, np.load(f"plates/labels_{TASK}_{split}.npz")


def targets(lab):
    cov = lab["covered"] if "covered" in lab.files else np.zeros(len(lab["present"]), np.int8)
    return (torch.tensor(lab["present"], dtype=torch.float32, device=dev), torch.tensor(lab["done"], dtype=torch.float32, device=dev),
            torch.tensor(lab["box"], dtype=torch.float32, device=dev), torch.tensor(cov, dtype=torch.float32, device=dev))


def run_eval(head, X, bs=256):
    head.eval()
    with torch.no_grad():
        o = torch.cat([head(X[i:i + bs].float()) for i in range(0, len(X), bs)])
    cov = torch.sigmoid(o[:, 6]) if o.shape[1] > 6 else torch.zeros_like(o[:, 0])
    return torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 1]), o[:, 2:6], cov


def iou(a, b):
    ix0, iy0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix1, iy1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def lossf(o, p, d, b, c):
    loss = (F.binary_cross_entropy_with_logits(o[:, 0], p) + F.binary_cross_entropy_with_logits(o[:, 1], d)
            + 5.0 * (F.smooth_l1_loss(o[:, 2:6], b, reduction="none").mean(1) * p).sum() / p.sum().clamp(min=1))
    if o.shape[1] > 6:
        loss = loss + F.binary_cross_entropy_with_logits(o[:, 6], c)
    return loss


def main():
    t0 = time.time()
    Xtr, ltr = load("train"); Xva, lva = load("val")
    Ptr, Dtr, Btr, Ctr = targets(ltr); Pva, Dva, Bva, Cva = targets(lva)
    print(f"loaded {len(Xtr)} train / {len(Xva)} val in {time.time() - t0:.0f}s", flush=True)
    head = (OL.BatteryHead() if TASK == "battery" else OL.Head()).to(dev)
    epochs, bs = 20, 128
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(Xtr) // bs), pct_start=0.1)
    best, best_state = 1e9, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = Xtr[idx].float()
            x = x + 0.10 * x.std() * torch.randn_like(x)
            x = x * (torch.rand(x.shape[0], 16, 16, 1, device=dev) > 0.12).float()   # patch-token dropout
            loss = lossf(head(x), Ptr[idx], Dtr[idx], Btr[idx], Ctr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        pp, dp, bp, cp = run_eval(head, Xva)
        vloss = (F.binary_cross_entropy(pp, Pva) + F.binary_cross_entropy(dp, Dva)
                 + 5.0 * (F.smooth_l1_loss(bp, Bva, reduction="none").mean(1) * Pva).sum() / Pva.sum()
                 + (F.binary_cross_entropy(cp, Cva) if TASK == "battery" else 0.0)).item()
        if vloss < best:
            best, best_state = vloss, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:2d} train_loss {loss.item():.3f} val_loss {vloss:.3f} best {best:.3f}", flush=True)
    head.load_state_dict(best_state)
    torch.save(best_state, f"{TASK}_head.pt")
    del Xtr
    Xte, lte = load("test")
    pp, dp, bp, cp = [t.cpu().numpy() for t in run_eval(head, Xte)]
    pres, done, box, kind = lte["present"], lte["done"], lte["box"], lte["kind"]
    covl = lte["covered"] if "covered" in lte.files else np.zeros(len(pres), np.int8)
    rep = {}
    pred_p = pp >= OL.PRESENT_THR
    rep["present_accuracy"] = float((pred_p == (pres == 1)).mean())
    rep["present_recall"] = float(pred_p[pres == 1].mean())
    rep["present_false_alarm_on_no_target"] = float(pred_p[pres == 0].mean())
    m = pres == 1
    rep["box_mean_iou"] = float(iou(bp[m], box[m]).mean())
    codes = np.array([OL.decide(TASK, float(pp[i]), float(dp[i]), tuple(bp[i]), float(cp[i]))[0] for i in range(len(pp))])
    gt = np.array([OL.decide(TASK, 1.0 if pres[i] else 0.0, 1.0 if done[i] else 0.0, tuple(box[i]))[0] for i in range(len(pp))])
    ok = codes == OL.OK
    rep["FALSE_OK_when_no_target"] = float(ok[pres == 0].mean())
    rep["FALSE_OK_when_incomplete_or_badly_framed"] = float(ok[(pres == 1) & (done == 0)].mean())
    rep["OK_recall_on_good_shots"] = float(ok[done == 1].mean())
    rep["OK_precision"] = float((done[ok] == 1).mean()) if ok.any() else None
    mv = np.isin(gt, [OL.LEFT, OL.RIGHT, OL.UP, OL.DOWN, OL.CLOSER, OL.BACK])
    rep["guidance_accuracy_when_a_move_is_needed"] = float((codes[mv] == gt[mv]).mean())
    kinds = [k for k in ["framed", "partial", "far_near", "covered", "negative"] if (kind == k).any()]
    rep["says_OK_by_kind"] = {k: float(ok[kind == k].mean()) for k in kinds}
    rep["guidance_matches_by_kind"] = {k: float((codes[kind == k] == gt[kind == k]).mean()) for k in kinds if k != "covered"}
    if TASK == "battery":
        rep["covered_detected_as_COVERED"] = float((codes[covl == 1] == OL.COVERED).mean())
        rep["bare_battery_wrongly_flagged_COVERED"] = float((codes[pres == 1] == OL.COVERED).mean())
    rep["predicted_code_counts"] = {OL.NAMES[c]: int((codes == c).sum()) for c in np.unique(codes)}
    rep["n_test"] = int(len(pp))
    json.dump(rep, open(f"{TASK}_eval.json", "w"), indent=1)
    print(json.dumps(rep, indent=1)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
