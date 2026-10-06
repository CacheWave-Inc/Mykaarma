"""How well does the present-probability separate real items from look-alikes in AUTO mode?"""
import numpy as np

import test_auto as T      # reuses call()/generators; its own report runs on import with N from argv

import synth_objects as SO
import synth_plates as SP

N = 250
true_p = {0: [], 1: [], 2: []}
false_p = {0: [], 1: [], 2: []}
for t, gen in ((0, lambda s: SO.make_clip("brake", s)), (1, lambda s: SO.make_clip("battery", s)), (2, SP.make_clip)):
    for seed in range(410000, 410000 + N):
        frames, lab = gen(seed)
        v = T.call(3, frames)
        d = int(v[9])
        if d < 0:
            continue
        (true_p if (lab["present"] and d == t) else false_p)[d].append(float(v[8]))
        # a plate-generator negative has no battery/wheel in it, so any detection there is false
for t in (0, 1, 2):
    a, b = np.array(true_p[t]), np.array(false_p[t])
    print(f"{T.NAME[t]:8s} true detections {len(a):3d} (p10={np.percentile(a, 10) if len(a) else 0:.3f} median={np.median(a) if len(a) else 0:.3f})   "
          f"other detections {len(b):3d} (median={np.median(b) if len(b) else 0:.3f} p90={np.percentile(b, 90) if len(b) else 0:.3f})")
for thr in (0.85, 0.95, 0.98, 0.99, 0.995):
    kept = {t: sum(p >= thr for p in true_p[t]) / max(1, len(true_p[t])) for t in (0, 1, 2)}
    fa = {t: sum(p >= thr for p in false_p[t]) for t in (0, 1, 2)}
    print(f"thr {thr}: true kept {{brake {kept[0]:.2f}, battery {kept[1]:.2f}, plate {kept[2]:.2f}}}  false remaining {{brake {fa[0]}, battery {fa[1]}, plate {fa[2]}}}")
