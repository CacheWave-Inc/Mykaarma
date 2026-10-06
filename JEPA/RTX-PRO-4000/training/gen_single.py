"""gen_single.py <task> : single-frame synthetic samples (JPEG bytes + labels) for the on-device model.
task in brake|battery|plate. Writes local/<task>_<split>.pkl"""
import multiprocessing as mp
import os
import pickle
import sys
import time

import cv2
import numpy as np

TASK = sys.argv[1]
SPLITS = {"train": (0, 16000), "val": (100000, 600), "test": (200000, 800)}


def gen(seed):
    import synth_objects
    import synth_plates
    frames, lab = synth_plates.make_clip(seed) if TASK == "plate" else synth_objects.make_clip(TASK, seed)
    f = frames[-1]
    ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return enc.tobytes(), int(lab["present"]), int(lab["done"]), np.array(lab["box"], np.float32), lab["kind"], int(lab.get("covered", 0)), np.array(lab.get("cbox", (0, 0, 0, 0)), np.float32)


def main():
    os.makedirs("local", exist_ok=True)
    pool = mp.Pool(15)
    for split, (start, count) in SPLITS.items():
        t0 = time.time()
        rows = pool.map(gen, range(start, start + count), chunksize=8)
        out = dict(jpg=[r[0] for r in rows], present=np.array([r[1] for r in rows], np.int8), done=np.array([r[2] for r in rows], np.int8),
                   box=np.stack([r[3] for r in rows]), kind=np.array([r[4] for r in rows]), covered=np.array([r[5] for r in rows], np.int8), cbox=np.stack([r[6] for r in rows]))
        pickle.dump(out, open(f"local/{TASK}_{split}.pkl", "wb"))
        print(f"{TASK} {split} {count} samples {time.time() - t0:.0f}s present={out['present'].mean():.2f} done={out['done'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
