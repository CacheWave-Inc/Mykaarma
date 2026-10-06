"""Triton Python-backend model `vjepa_inspect`: ONE frozen V-JEPA 2 encoder + a trained head per inspection task.

FRAMES = [task:1 (bit 7 = verify a still photo, more tolerant)][n:1] + n x ([len:4 BE][JPEG 256x256]);  task 0 = brake, 1 = battery, 2 = license plate, 3 = AUTO (any of the three).
GUIDANCE (FP32, 12) = [code, ready, dx, dy, size, brightness, sharpness, motion, present_prob, done_prob, infer_ms, frames_used]
`code` uses the plate_logic constants; NO_PLATE (13) means "target not in view" for every task.
"""
import os
import struct
import sys
import time

import cv2
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obj_logic as OL  # noqa: E402
import plate_logic as PL  # noqa: E402

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MOTION_MAX = 12.0
SHARP_MIN = {0: 15.0, 1: 15.0, 2: 20.0}
TASKS = {0: "brake", 1: "battery", 2: "plate"}
BOX_SCALE = {"battery": 0.45}
AUTO = 3          # task byte 3: look for any of the three items; v[9] then carries the detected task (-1 = nothing found)


def parse(data):
    n = data[1]
    off, frames = 2, []
    for _ in range(n):
        (length,) = struct.unpack(">I", data[off:off + 4])
        off += 4
        img = cv2.imdecode(np.frombuffer(data[off:off + length], np.uint8), cv2.IMREAD_COLOR)
        off += length
        if img is None:
            continue
        if img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


class TritonPythonModel:
    def initialize(self, args):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
        here = os.path.dirname(os.path.abspath(__file__))
        self.heads = {}
        for t, name in TASKS.items():
            h = (OL.BatteryHead() if name == "battery" else PL.Head()).cuda().eval()
            h.load_state_dict(torch.load(os.path.join(here, f"{name}_head.pt"), map_location="cuda"))
            self.heads[t] = h
        print("[vjepa_inspect] encoder + heads loaded:", list(TASKS.values()), flush=True)

    def _score(self, task, frames, verify=False):
        out = np.zeros(12, dtype=np.float32)
        n = min(len(frames), 16)
        n -= n % 2
        if n < 4:
            out[0], out[11] = PL.WARMING, n
            return out
        frames = frames[-n:]
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        brightness = float(gray[-1].mean())
        small = [cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) for g in gray[-4:]]
        motion = float(np.mean([np.abs(small[i + 1] - small[i]).mean() for i in range(len(small) - 1)]))

        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32) / 255.0
        clip = ((rgb - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float16)
        x = torch.from_numpy(clip).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            h = self.model(pixel_values_videos=x).last_hidden_state[0].float()
            tok = h.view(n // 2, 16, 16, -1).mean(0, keepdim=True)
            outs = {t: self.heads[t](tok)[0].cpu().numpy() for t in (TASKS if task == AUTO else [task])}
        torch.cuda.synchronize()
        infer_ms = (time.time() - t0) * 1000.0

        sig = lambda v: float(1 / (1 + np.exp(-v)))
        detected = -1
        if task == AUTO:
            best, best_s = None, 0.0
            for t, ot in outs.items():
                p, cp = sig(ot[0]), (sig(ot[6]) if len(ot) > 6 else 0.0)
                sc = p if p >= PL.PRESENT_THR else 0.0          # 'covered battery' is only reported when the battery task is chosen explicitly
                if sc > best_s:
                    best, best_s = t, sc
            if best is None:                                    # nothing recognisable in view
                sharp = float(cv2.Laplacian(gray[-1], cv2.CV_64F).var())
                code = PL.DARK if brightness < 35 else PL.BRIGHT if brightness > 225 else PL.NO_PLATE
                out[:] = [code, 0.0, 0, 0, 0, brightness, sharp, motion, 0.0, -1.0, infer_ms, n]
                return out
            task = detected = best
        o = outs[task]
        present_p = float(1 / (1 + np.exp(-o[0])))
        done_p = float(1 / (1 + np.exp(-o[1])))
        covered_p = float(1 / (1 + np.exp(-o[6]))) if len(o) > 6 else 0.0
        box = tuple(float(v) for v in o[2:6])
        # STOPGAP: on real photos the battery head (trained on synthetic batteries only) overestimates the box about 2.5x
        # (measured on two real views). Shrink it about its centre until the head is fine-tuned on real photos.
        sc = BOX_SCALE.get(TASKS[task], 1.0)
        if sc != 1.0:
            cx_, cy_ = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            hw_, hh_ = (box[2] - box[0]) * sc / 2, (box[3] - box[1]) * sc / 2
            box = (cx_ - hw_, cy_ - hh_, cx_ + hw_, cy_ + hh_)
        # verify=True re-checks a still photo the live check already accepted: allow ~10% overshoot at the edges and a
        # softer 'complete' probability, so a borderline frame does not bounce between OK and not-OK
        kw = dict(margin=-0.10, done_thr=0.0, present_thr=0.40) if verify else {}      # the 'complete' score is not meaningful for a still: only presence, framing and size are checked
        if task == 2:
            code, dx, dy, size = PL.decide(present_p, done_p, box, **kw)
        else:
            code, dx, dy, size = OL.decide(TASKS[task], present_p, done_p, box, covered_p, **kw)

        x0, y0, x1, y1 = [int(np.clip(v, 0, 1) * 256) for v in box]
        found = present_p >= (0.40 if verify else PL.PRESENT_THR) and x1 - x0 > 8 and y1 - y0 > 8
        region = gray[-1][y0:y1, x0:x1] if found else gray[-1]
        sharp = float(cv2.Laplacian(region, cv2.CV_64F).var())
        # plates are judged on the plate itself; dark objects (black battery, tyre) on the whole frame
        light = float(gray[-1][y0:y1, x0:x1].mean()) if (found and task == 2) else brightness
        dark_min = 45 if task == 2 else 35
        if light < dark_min:
            code = PL.DARK
        elif light > 225:
            code = PL.BRIGHT
        elif code == PL.OK and motion > MOTION_MAX and not verify:
            code = PL.SHAKY
        elif code == PL.OK and sharp < SHARP_MIN[task] * (0.5 if verify else 1.0):
            code = PL.BLURRY

        out[:] = [code, 1.0 if code == PL.OK else 0.0, dx, dy, size, brightness, sharp, motion,
                  present_p, float(detected) if detected >= 0 else done_p, infer_ms, n]
        return out

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                data = pb_utils.get_input_tensor_by_name(request, "FRAMES").as_numpy().tobytes()
                verify = bool(data[0] & 0x80)
                task = (data[0] & 0x7F) if ((data[0] & 0x7F) in TASKS or (data[0] & 0x7F) == AUTO) else 2
                out = self._score(task, parse(data), verify)
            except Exception as exc:
                print(f"[vjepa_inspect] error: {exc!r}", flush=True)
                out = np.zeros(12, dtype=np.float32)
                out[0] = PL.ERROR
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("GUIDANCE", out)]))
        return responses

    def finalize(self):
        pass
