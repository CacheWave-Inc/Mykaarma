"""build_features_obj.py <task> [splits...]  -> plates/feats_<task>_<split>.npy + labels (frozen V-JEPA 2 time-mean tokens)."""
import multiprocessing as mp
import sys
import time

import numpy as np

TASK = sys.argv[1]
SPLITS = {"train": (0, 14000), "val": (100000, 500), "test": (200000, 700)}
ONLY = set(sys.argv[2:])


def gen(seed):
    import synth_objects
    return synth_objects.make_clip(TASK, seed)


def main():
    pool = mp.Pool(12)
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)
    for split, (start, count) in SPLITS.items():
        if ONLY and split not in ONLY:
            continue
        feats = np.lib.format.open_memmap(f"plates/feats_{TASK}_{split}.npy", mode="w+", dtype=np.float16, shape=(count, 16, 16, 1024))
        present = np.zeros(count, np.int8); cov = np.zeros(count, np.int8); done = np.zeros(count, np.int8)
        box = np.zeros((count, 4), np.float32); nfr = np.zeros(count, np.int8); kind = np.zeros(count, "U10")
        t0 = time.time()
        for i, (frames, lab) in enumerate(pool.imap(gen, range(start, start + count), chunksize=4)):
            x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
            x = ((x - mean) / std).half()
            with torch.no_grad():
                h = model(pixel_values_videos=x).last_hidden_state[0]
            feats[i] = h.view(frames.shape[0] // 2, 16, 16, -1).float().mean(0).half().cpu().numpy()
            present[i], done[i], box[i], nfr[i], kind[i] = lab["present"], lab["done"], lab["box"], lab["n"], lab["kind"]
            cov[i] = lab.get("covered", 0)
            if i % 1000 == 0:
                print(f"{TASK} {split} {i}/{count} {time.time() - t0:.0f}s", flush=True)
        feats.flush()
        np.savez(f"plates/labels_{TASK}_{split}.npz", present=present, done=done, box=box, n=nfr, kind=kind, covered=cov)
        print(f"{TASK} {split} done: present={present.mean():.2f} done={done.mean():.2f} {time.time() - t0:.0f}s", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
