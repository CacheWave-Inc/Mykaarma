"""Shared by training/eval and the Triton model: head definition + box -> guidance code."""
import torch
import torch.nn as nn

OK, DARK, BRIGHT, BLURRY, SHAKY, LEFT, RIGHT, UP, DOWN, CLOSER, BACK, WARMING, ERROR, NO_PLATE, ADJUST = range(15)
NAMES = "OK DARK BRIGHT BLURRY SHAKY LEFT RIGHT UP DOWN CLOSER BACK WARMING ERROR NO_PLATE ADJUST".split()

MARGIN = 0.03          # plate must sit this far inside every frame edge
MIN_W, MAX_W = 0.28, 0.88
PRESENT_THR, DONE_THR = 0.85, 0.95


class Head(nn.Module):
    """Frozen V-JEPA 2 time-mean patch tokens [B,16,16,1024] -> present, done, box(4)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(0.1), nn.Linear(1024, 256), nn.GELU())
        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GELU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 16, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 6))

    def forward(self, tok):
        x = self.proj(tok).permute(0, 3, 1, 2)
        return self.fc(self.conv(x))   # [B,6]: present_logit, done_logit, x0, y0, x1, y1


def decide(present_p, done_p, box, margin=MARGIN, done_thr=DONE_THR, present_thr=PRESENT_THR):
    """Return (code, dx, dy, width) from head outputs. box = (x0,y0,x1,y1) normalised, may exceed [0,1]."""
    if present_p < present_thr:
        return NO_PLATE, 0.0, 0.0, 0.0
    x0, y0, x1, y1 = box
    w = x1 - x0
    dx, dy = (x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5
    if w > 0.95:
        return BACK, dx, dy, w
    over = {LEFT: margin - x0, RIGHT: x1 - (1 - margin), UP: margin - y0, DOWN: y1 - (1 - margin)}
    code, amount = max(over.items(), key=lambda kv: kv[1])
    if amount > 0:
        return code, dx, dy, w
    if w < MIN_W:
        return CLOSER, dx, dy, w
    if w > MAX_W:
        return BACK, dx, dy, w
    if done_p >= done_thr:
        return OK, dx, dy, w
    return ADJUST, dx, dy, w
