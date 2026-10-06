"""Auto-label real frames with an open-vocabulary detector (OWLv2): best box per query per frame.
usage: label_owl.py <frames_dir> <out_json> [sheet.png]"""
import glob
import json
import os
import sys

import torch
from PIL import Image, ImageDraw
from transformers import Owlv2ForObjectDetection, Owlv2Processor

frames_dir, out_json = sys.argv[1], sys.argv[2]
sheet_png = sys.argv[3] if len(sys.argv) > 3 else None
QUERIES = ["a car battery", "a license plate", "a car wheel with brake disc"]
NAMES = ["battery", "plate", "brake"]
MODEL = "google/owlv2-base-patch16-ensemble"

proc = Owlv2Processor.from_pretrained(MODEL)
model = Owlv2ForObjectDetection.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
res = {}
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    inp = proc(text=[QUERIES], images=im, return_tensors="pt")
    inp = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_grounded_object_detection(out, threshold=0.0, target_sizes=torch.tensor([im.size[::-1]]).cuda())[0]
    best = {}
    for s, l, b in zip(r["scores"].float().cpu().tolist(), r["labels"].cpu().tolist(), r["boxes"].float().cpu().tolist()):
        if NAMES[l] not in best or s > best[NAMES[l]][0]:
            best[NAMES[l]] = (s, [round(v, 1) for v in b])
    res[os.path.basename(f)] = {k: {"score": round(v[0], 3), "box": v[1]} for k, v in best.items()}
    if i % 50 == 0:
        print(i, len(files), flush=True)
json.dump(res, open(out_json, "w"))
if sheet_png:
    step = max(1, len(files) // 24)
    pick = files[::step][:24]
    sheet = Image.new("RGB", (6 * 256, 4 * 256), "white")
    for k, f in enumerate(pick):
        im = Image.open(f).convert("RGB"); d = ImageDraw.Draw(im)
        for name, col in (("battery", (0, 255, 0)), ("plate", (255, 160, 0)), ("brake", (0, 160, 255))):
            e = res[os.path.basename(f)].get(name)
            if e and e["score"] >= 0.20:
                d.rectangle(e["box"], outline=col, width=4)
                d.text((e["box"][0] + 4, e["box"][1] + 4), f"{name} {e['score']:.2f}", fill=col)
        sheet.paste(im.resize((256, 256)), ((k % 6) * 256, (k // 6) * 256))
    sheet.save(sheet_png)
print("done", len(res))
