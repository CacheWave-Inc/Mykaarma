import numpy as np, torch
import plate_logic as PL
from train_head import load, run_eval
head = PL.Head().cuda(); head.load_state_dict(torch.load("plate_head.pt")); head.eval()
X, lab = load("test")
pp, dp, bp = [t.cpu().numpy() for t in run_eval(head, X, None)]
pres, done, kind = lab["present"], lab["done"], lab["kind"]
print("present_thr done_thr | FALSE_OK no-plate | FALSE_OK incomplete | OK recall on good shots")
for pt in (0.5, 0.7, 0.85):
    for dt in (0.6, 0.8, 0.9, 0.95):
        PL.PRESENT_THR, PL.DONE_THR = pt, dt
        ok = np.array([PL.decide(float(pp[i]), float(dp[i]), tuple(bp[i]))[0] == PL.OK for i in range(len(pp))])
        print(f"   {pt:.2f}       {dt:.2f}   |   {ok[pres==0].mean()*100:5.2f}%   |   {ok[(pres==1)&(done==0)].mean()*100:5.2f}%   |   {ok[done==1].mean()*100:5.1f}%")
