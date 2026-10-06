"""Which false detections does AUTO mode make on scenes with no target, and through which path?
Negatives come from the plate generator (signs, blank plates, empty cars, text, scenes: none contains a battery or a wheel)."""
import collections
import sys

import numpy as np

sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "triton", "1"]
import test_auto as T            # noqa: E402  (prints its own small report with N=1)

import synth_plates as SP        # noqa: E402

cnt = collections.Counter()
P = collections.defaultdict(list)
n = 0
for seed in range(420000, 420500):
    frames, lab = SP.make_clip(seed)
    if lab["present"] or lab["kind"] != "negative":
        continue
    v = T.call(3, frames)
    n += 1
    d = int(v[9])
    if d < 0:
        cnt["nothing"] += 1
    else:
        via = "covered-path" if int(v[0]) == 15 else "normal"
        cnt[f"{T.NAME[d]} ({via})"] += 1
        P[f"{T.NAME[d]} ({via})"].append(float(v[8]))
print("negatives:", n, dict(cnt))
