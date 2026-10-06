import sys, numpy as np, cv2
from synth_plates import make_clip, S
rows, cols = 5, 8
kinds = ["framed", "partial", "partial", "far_near", "negative"]
tiles = []
for r in range(rows):
    for c in range(cols):
        frames, lab = make_clip(1000 + r * cols + c, force=kinds[r])
        im = frames[-1][..., ::-1].copy()
        if lab["present"]:
            x0, y0, x1, y1 = [int(v * S) for v in lab["box"]]
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0) if lab["done"] else (0, 165, 255), 2)
        cv2.putText(im, f"P{lab['present']} D{lab['done']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(cv2.resize(im, (160, 160)))
grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
cv2.imwrite("preview.png", grid)
print(grid.shape)
