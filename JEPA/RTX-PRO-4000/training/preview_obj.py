import sys
import cv2
import numpy as np
import synth_objects as SO

task = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
tiles = []
for i in range(32):
    frames, lab = SO.make_clip(task, start + i)
    im = np.ascontiguousarray(frames[-1][..., ::-1])
    b = lab["box"]
    if lab["present"]:
        cv2.rectangle(im, (int(b[0] * 256), int(b[1] * 256)), (int(b[2] * 256), int(b[3] * 256)), (0, 255, 0) if lab["done"] else (0, 165, 255), 2)
    cv2.putText(im, f'{lab["kind"][:6]} p{lab["present"]} d{lab["done"]}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(im)
rows = [np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(4)]
cv2.imwrite(f"preview_{task}.png", np.vstack(rows))
print("ok")
