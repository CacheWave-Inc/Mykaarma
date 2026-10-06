"""eval_thr.py <task>: how the 'complete' (done) threshold trades false-OKs against missed-OKs on held-out synthetic clips."""
import sys

import numpy as np
import torch

import obj_logic as OL

TASK = sys.argv[1]
dev = "cuda"
head = (OL.BatteryHead() if TASK == "battery" else OL.Head()).to(dev)
head.load_state_dict(torch.load(f"{TASK}_head.pt", map_location=dev))
head.eval()
f = np.load(f"plates/feats_{TASK}_test.npy", mmap_mode="r")
lab = np.load(f"plates/labels_{TASK}_test.npz")
X = torch.from_numpy(np.ascontiguousarray(f[:])).to(dev)
with torch.no_grad():
    o = torch.cat([head(X[i:i + 128].float()) for i in range(0, len(X), 128)]).cpu().numpy()
sig = lambda v: 1 / (1 + np.exp(-v))
pp, dp, bp = sig(o[:, 0]), sig(o[:, 1]), o[:, 2:6]
cp = sig(o[:, 6]) if o.shape[1] > 6 else np.zeros(len(pp))
pres, done = lab["present"], lab["done"]
print(f"{TASK}: n={len(pp)}  done_thr -> OK on good | false OK on incomplete/bad | false OK with no target")
for thr in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35):
    codes = np.array([OL.decide(TASK, pp[i], dp[i], tuple(bp[i]), cp[i], done_thr=thr)[0] for i in range(len(pp))])
    ok = codes == OL.OK
    print(f"  {thr:4.2f}: {ok[done == 1].mean():.3f} | {ok[(pres == 1) & (done == 0)].mean():.3f} | {ok[pres == 0].mean():.3f}")
