"""Guidance decision for the battery / brake heads (same Head as plates; size = longest box side)."""
import torch.nn as nn

from plate_logic import (Head, OK, DARK, BRIGHT, BLURRY, SHAKY, LEFT, RIGHT, UP, DOWN, CLOSER, BACK, WARMING, ERROR,  # noqa: F401
                         NO_PLATE, ADJUST, MARGIN, PRESENT_THR, DONE_THR)
from plate_logic import NAMES as _PLATE_NAMES

COVERED = 15
NAMES = _PLATE_NAMES + ["COVERED"]
SIZE = {"battery": (0.30, 0.88), "brake": (0.38, 0.92)}
COVERED_THR = 0.85
DONE = {"battery": 0.0, "brake": 0.6}      # battery: the head's "complete" score is ~0 on real photos, so only presence, size and framing decide   # live "complete" threshold; held-out sweep: false-OKs barely move below 0.95, real photos are less confident


class BatteryHead(Head):
    """Plate head + one extra logit: 'a battery is there but it is still covered / boxed in'."""

    def __init__(self):
        super().__init__()
        self.fc[4] = nn.Linear(512, 7)


def decide(task, present_p, done_p, box, covered_p=0.0, margin=MARGIN, done_thr=None, present_thr=PRESENT_THR):
    """-> (code, dx, dy, size). NO_PLATE doubles as 'target not in view' for every task."""
    if present_p < present_thr:
        if task == "battery" and covered_p >= COVERED_THR:
            return COVERED, 0.0, 0.0, 0.0
        return NO_PLATE, 0.0, 0.0, 0.0
    x0, y0, x1, y1 = box
    size = max(x1 - x0, y1 - y0)
    dx, dy = (x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5
    lo, hi = SIZE[task]
    if done_thr is None:
        done_thr = DONE[task]
    if size > 1.0:
        return BACK, dx, dy, size
    over = {LEFT: margin - x0, RIGHT: x1 - (1 - margin), UP: margin - y0, DOWN: y1 - (1 - margin)}
    code, amount = max(over.items(), key=lambda kv: kv[1])
    if amount > 0:
        return code, dx, dy, size
    if size < lo:
        return CLOSER, dx, dy, size
    if size > hi:
        return BACK, dx, dy, size
    if done_p >= done_thr:
        return OK, dx, dy, size
    return ADJUST, dx, dy, size
