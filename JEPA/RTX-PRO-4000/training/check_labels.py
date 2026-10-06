import pickle
import sys

import cv2
import numpy as np

task = sys.argv[1]
d = pickle.load(open(f"local/{task}_val.pkl", "rb"))
tiles = []
for i in range(32):
    im = cv2.imdecode(np.frombuffer(d["jpg"][i], np.uint8), cv2.IMREAD_COLOR)
    if d["present"][i]:
        b = np.clip(d["box"][i], 0, 1) * 256
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    if d["covered"][i]:
        b = np.clip(d["cbox"][i], 0, 1) * 256
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 2)
    cv2.putText(im, str(d["kind"][i])[:7], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(im)
cv2.imwrite(f"labels_{task}.png", np.vstack([np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(4)]))
p = d["present"] == 1
print("present", int(p.sum()), "covered", int(d["covered"].sum()), "of", len(p))
w = np.clip(d["box"][p][:, 2], 0, 1) - np.clip(d["box"][p][:, 0], 0, 1)
h = np.clip(d["box"][p][:, 3], 0, 1) - np.clip(d["box"][p][:, 1], 0, 1)
print("bare box width quantiles", np.round(np.quantile(w, [0.1, 0.5, 0.9]), 2), "height", np.round(np.quantile(h, [0.1, 0.5, 0.9]), 2))
print("mean image brightness", float(np.mean([cv2.imdecode(np.frombuffer(j, np.uint8), 0).mean() for j in d["jpg"][:200]])))
