"""Decision rules on a detected box (shared by training evaluation and the RT-DETR server).
Codes match plate_logic / obj_logic. Boxes are (x0, y0, x1, y1) normalised and clipped to the frame, because a detector cannot
output a box outside the image: an object cut off by the frame shows up as a box touching that edge."""
OK, DARK, BRIGHT, BLURRY, SHAKY, LEFT, RIGHT, UP, DOWN, CLOSER, BACK, WARMING, ERROR, NO_PLATE, ADJUST, COVERED = range(16)
NAMES = "OK DARK BRIGHT BLURRY SHAKY LEFT RIGHT UP DOWN CLOSER BACK WARMING ERROR NO_PLATE ADJUST COVERED".split()

MARGIN = 0.03
SIZE = {"battery": (0.30, 0.88), "brake": (0.38, 0.92), "plate": (0.28, 0.88)}   # plate: width; others: longest side


def decide(task, score, box, cov_score=0.0, thr=0.5, margin=MARGIN, gt_mode=False):
    """-> code. In gt_mode `score` is a stand-in for presence/done of a ground-truth box (>=0.95 good, >=0.5 present)."""
    if score < (0.5 if gt_mode else thr):
        if task == "battery" and cov_score >= thr and not gt_mode:
            return COVERED
        return NO_PLATE
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = x1 - x0, y1 - y0
    size = w if task == "plate" else max(w, h)
    over = {LEFT: margin - x0, RIGHT: x1 - (1 - margin), UP: margin - y0, DOWN: y1 - (1 - margin)}
    code, amount = max(over.items(), key=lambda kv: kv[1])
    if amount > 0:
        return code
    lo, hi = SIZE[task]
    if size < lo:
        return CLOSER
    if size > hi:
        return BACK
    if gt_mode and score < 0.95:
        return ADJUST
    return OK
