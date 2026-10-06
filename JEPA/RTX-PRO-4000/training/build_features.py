"""Generate synthetic plate clips and cache frozen V-JEPA 2 features (time-mean patch tokens).

Output per split: plates/feats_<split>.f16 (memmap [N,16,16,1024] fp16) + plates/labels_<split>.npz
"""
import multiprocessing as mp
import sys
import time

import numpy as np

SPLITS = {"train": (0, 8000), "val": (100000, 1000), "test": (200000, 1000), "train2": (8000, 12000)}
ONLY = set(sys.argv[1:])


def gen(seed):
    from synth_plates import make_clip
    frames, lab = make_clip(seed)
    return frames, lab


def main():
    pool = mp.Pool(12)          # fork workers before CUDA is initialised
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)

    for split, (start, count) in SPLITS.items():
        if ONLY and split not in ONLY:
            continue
        feats = np.lib.format.open_memmap(f"plates/feats_{split}.npy", mode="w+", dtype=np.float16, shape=(count, 16, 16, 1024))
        present = np.zeros(count, np.int8); done = np.zeros(count, np.int8)
        box = np.zeros((count, 4), np.float32); nfr = np.zeros(count, np.int8); kind = np.zeros(count, "U10")
        t0 = time.time()
        for i, (frames, lab) in enumerate(pool.imap(gen, range(start, start + count), chunksize=4)):
            x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
            x = ((x - mean) / std).half()
            with torch.no_grad():
                h = model(pixel_values_videos=x).last_hidden_state[0]
            tok = h.view(frames.shape[0] // 2, 16, 16, -1).float().mean(0)
            feats[i] = tok.half().cpu().numpy()
            present[i], done[i], box[i], nfr[i], kind[i] = lab["present"], lab["done"], lab["box"], lab["n"], lab["kind"]
            if i % 500 == 0:
                print(f"{split} {i}/{count}  {time.time() - t0:.0f}s", flush=True)
        feats.flush()
        np.savez(f"plates/labels_{split}.npz", present=present, done=done, box=box, n=nfr, kind=kind)
        print(f"{split} done: present={present.mean():.2f} done={done.mean():.2f}  {time.time() - t0:.0f}s", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
