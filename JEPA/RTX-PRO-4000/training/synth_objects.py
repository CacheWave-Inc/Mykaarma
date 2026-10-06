"""Synthetic service-centre scenes for BATTERY and BRAKE (wheel) guidance, same label contract as synth_plates.

make_clip(task, seed) -> (frames uint8 [n,256,256,3] RGB, labels dict)   task in {"battery", "brake"}
labels: present (target meaningfully visible), done (complete + well framed), box = (x0,y0,x1,y1) of the FULL target
        in [0,1] frame coords (may lie outside [0,1]), kind, n.
"""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

import synth_plates as SP
from synth_plates import S, FONT_FILES, font

BRANDS = ["POWERMAX", "DURACELL", "EXIDE", "BOSCH", "ACDELCO", "OPTIMA", "VARTA", "INTERSTATE", "MOTORCRAFT", "DIEHARD",
          "YUASA", "ODYSSEY", "NAPA", "CENTURY", "WESTCO", "STARTMAX"]
CALIPER_TXT = ["BREMBO", "AKEBONO", "BRAKE", "TRW", "SPORT", "DISC", "ATE", "ZIMMER"]
CAL_COL = [(190, 25, 30), (230, 190, 20), (30, 70, 170), (25, 25, 28), (170, 172, 178), (230, 120, 20), (40, 130, 70)]
RIM_COL = [(185, 188, 195), (30, 30, 34), (120, 100, 70), (95, 98, 105), (215, 215, 220), (60, 60, 70)]


# ----------------------------------------------------------------------------------------------------- battery
def battery_texture(rng):
    """BARE 12 V battery (no cover): top face with terminals/caps, optionally the front face with the label.
    -> tex, aspect, terminals [(x, y, positive)]"""
    W = rng.randint(520, 760)
    asp = rng.uniform(1.35, 2.3)
    Ht = int(W / asp)
    front = rng.random() < 0.45
    Hf = int(Ht * rng.uniform(0.4, 0.65)) if front else 0
    H = Ht + Hf
    body = rng.choice([(26, 26, 30), (44, 44, 50), (62, 64, 68), (205, 205, 208), (24, 48, 108), (118, 26, 26), (36, 62, 42), (70, 70, 74)])
    dark = sum(body) < 300
    img = Image.new("RGB", (W, H), tuple(int(c * 0.7) for c in body))
    d = ImageDraw.Draw(img)
    m = int(Ht * 0.05)
    d.rectangle([m, m, W - m, Ht - m], fill=body)
    for i in range(rng.randint(2, 6)):                                    # moulded ribs
        y = int(Ht * rng.uniform(0.2, 0.9))
        d.line([m, y, W - m, y], fill=tuple(int(c * (1.15 if dark else 0.88)) for c in body), width=rng.randint(1, 3))
    hi = tuple(min(255, int(c * 1.3) + 12) for c in body) if dark else tuple(int(c * 0.82) for c in body)
    if rng.random() < 0.7:                                                # carry handle strap
        d.rounded_rectangle([int(W * 0.2), int(Ht * 0.04), int(W * 0.8), int(Ht * 0.16)], radius=8, fill=hi)
    if front:                                                             # front face darker, carries the label
        fb = tuple(int(c * 0.85) for c in body)
        d.rectangle([m, Ht, W - m, H - m], fill=fb)
    lab = rng.choice([(235, 235, 235), (240, 200, 40), (190, 30, 30), (30, 90, 190), (240, 240, 230), (20, 20, 20), (30, 140, 70)])
    if front:
        lx0, lx1, ly0, ly1 = int(W * 0.1), int(W * 0.9), Ht + int(Hf * 0.12), H - int(Hf * 0.2)
    else:
        lx0, lx1 = int(W * rng.uniform(0.12, 0.2)), int(W * rng.uniform(0.8, 0.88))
        ly0, ly1 = int(Ht * rng.uniform(0.42, 0.5)), int(Ht * rng.uniform(0.82, 0.92))
    d.rectangle([lx0, ly0, lx1, ly1], fill=lab)
    lfg = (20, 20, 20) if sum(lab) > 330 else (240, 240, 240)
    ff = rng.choice(FONT_FILES)
    lh = ly1 - ly0
    d.text(((lx0 + lx1) // 2, ly0 + int(lh * 0.3)), rng.choice(BRANDS), fill=lfg, font=font(ff, max(int(lh * 0.34), 12)), anchor="mm")
    d.text(((lx0 + lx1) // 2, ly0 + int(lh * 0.72)),
           f"12V  {rng.choice([400, 500, 600, 650, 700, 800])} CCA   {rng.choice(['H6', 'H7', '24F', '35', '65', 'T5'])}",
           fill=lfg, font=font(ff, max(int(lh * 0.2), 10)), anchor="mm")
    for k in range(rng.choice([0, 3, 6])):                                # vent caps
        cx = int(W * (0.25 + 0.5 * (k % 3) / 2)); cy = int(Ht * (0.28 + 0.10 * (k // 3)))
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=hi, outline=tuple(int(c * 0.6) for c in body))
    r = int(Ht * rng.uniform(0.075, 0.1))                                 # terminals
    ty = int(Ht * rng.uniform(0.18, 0.28))
    pos_left = rng.random() < 0.5
    tp, tn = (int(W * 0.17), int(W * 0.83)) if pos_left else (int(W * 0.83), int(W * 0.17))
    terms = []
    for tx, positive in ((tp, True), (tn, False)):
        rr = int(r * (1.0 if positive else 0.82))
        d.ellipse([tx - rr - 4, ty - rr - 4, tx + rr + 4, ty + rr + 4], fill=(60, 60, 64))
        d.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], fill=(176, 178, 184), outline=(110, 110, 116), width=3)
        d.ellipse([tx - rr // 2, ty - rr // 2, tx + rr // 2, ty + rr // 2], fill=(130, 132, 138))
        if rng.random() < 0.8:                                            # terminal clamp with bolt
            cap = (200, 30, 30) if (positive and rng.random() < 0.7) else (25, 25, 28) if rng.random() < 0.7 else (150, 152, 158)
            d.rectangle([tx - rr - 12, ty - rr * 0.5, tx + rr + 12, ty + rr * 0.5], fill=cap)
            d.ellipse([tx + rr - 4, ty - 6, tx + rr + 8, ty + 6], fill=(200, 200, 205))
        if rng.random() < 0.25:                                           # corrosion powder
            for _ in range(rng.randint(6, 18)):
                px, py = tx + rng.randint(-rr - 8, rr + 8), ty + rng.randint(-rr - 8, rr + 8)
                d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=rng.choice([(235, 240, 235), (150, 200, 180), (210, 215, 205)]))
        d.text((tx, ty - int(rr * 1.9)), "+" if positive else "-", fill=(235, 235, 235) if dark else (30, 30, 30),
               font=font(FONT_FILES[0], int(Ht * 0.12)), anchor="mm")
        terms.append((tx, ty, positive))
    if rng.random() < 0.4:                                                # warning sticker
        d.polygon([(int(W * 0.5), int(Ht * 0.2)), (int(W * 0.46), int(Ht * 0.33)), (int(W * 0.54), int(Ht * 0.33))], fill=(240, 200, 30))
    arr = np.asarray(img).astype(np.float32)
    gy = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.1), H, dtype=np.float32)[:, None, None]
    arr = arr * gy + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 3, arr.shape)
    if rng.random() < 0.5:                                                # specular glare
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        gx0, gy0, sg = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.08, 0.25) * W
        arr += (np.exp(-((xx - gx0) ** 2 + (yy - gy0) ** 2) / (2 * sg ** 2)) * rng.uniform(40, 110))[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8), W / H, terms


def cover_texture(rng):
    """Battery WITH its cover / box on: same footprint, closed plastic lid or heat-shield, no terminals or label visible."""
    W = rng.randint(520, 760)
    H = int(W / rng.uniform(1.3, 2.3))
    body = rng.choice([(24, 24, 28), (40, 40, 44), (58, 58, 62), (90, 92, 96), (180, 180, 184), (30, 40, 70)])
    dark = sum(body) < 300
    img = Image.new("RGB", (W, H), tuple(int(c * 0.7) for c in body))
    d = ImageDraw.Draw(img)
    m = int(H * 0.04)
    d.rectangle([m, m, W - m, H - m], fill=body)
    style = rng.random()
    if style < 0.35:                                                      # quilted heat shield
        for x in range(m, W - m, rng.randint(36, 70)):
            d.line([x, m, x, H - m], fill=tuple(int(c * 0.7) for c in body), width=3)
        for y in range(m, H - m, rng.randint(36, 70)):
            d.line([m, y, W - m, y], fill=tuple(int(c * 0.7) for c in body), width=3)
    elif style < 0.7:                                                     # moulded plastic lid with ribs
        for y in range(m + 20, H - m, rng.randint(28, 55)):
            d.line([m + 10, y, W - m - 10, y], fill=tuple(int(c * (1.2 if dark else 0.85)) for c in body), width=rng.randint(2, 5))
        d.rectangle([int(W * 0.15), int(H * 0.2), int(W * 0.85), int(H * 0.8)], outline=tuple(int(c * 0.6) for c in body), width=4)
    else:                                                                 # enclosed battery box with strap
        d.rectangle([int(W * 0.05), int(H * 0.44), int(W * 0.95), int(H * 0.56)], fill=(70, 72, 78))
        d.rectangle([int(W * 0.46), int(H * 0.40), int(W * 0.54), int(H * 0.60)], fill=(190, 190, 196))
    for sx in (int(W * 0.06), int(W * 0.94)):                             # latch clips
        d.rectangle([sx - 10, int(H * 0.3), sx + 10, int(H * 0.7)], fill=(60, 60, 64))
    if rng.random() < 0.6:                                                # cables leaving through grommets
        for gx, col in ((int(W * 0.2), (190, 25, 25)), (int(W * 0.8), (22, 22, 24))):
            d.ellipse([gx - 12, int(H * 0.08) - 12, gx + 12, int(H * 0.08) + 12], fill=(10, 10, 12))
            d.line([gx, int(H * 0.08), gx + rng.randint(-40, 40), 0], fill=col, width=8)
    if rng.random() < 0.5:
        d.text((W // 2, int(H * 0.5)), rng.choice(["BATTERY", "12V", "WARNING", "CAUTION", "BATT"]),
               fill=(235, 235, 235) if dark else (30, 30, 30), font=font(rng.choice(FONT_FILES), int(H * 0.16)), anchor="mm")
    arr = np.asarray(img).astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 3, (H, W, 3))
    return np.clip(arr, 0, 255).astype(np.uint8), W / H


def container_texture(rng):
    """Hard negative: battery-like box WITHOUT terminals (air box, fuse box, coolant tank, toolbox)."""
    W = rng.randint(500, 720)
    H = int(W / rng.uniform(1.2, 2.2))
    body = rng.choice([(26, 26, 30), (50, 50, 55), (190, 190, 195), (220, 215, 150), (40, 70, 120), (150, 40, 40)])
    img = Image.new("RGB", (W, H), tuple(int(c * 0.8) for c in body))
    d = ImageDraw.Draw(img)
    m = int(H * 0.05)
    d.rectangle([m, m, W - m, H - m], fill=body)
    for x in range(int(W * 0.1), int(W * 0.95), rng.randint(40, 90)):    # ribs / grille
        d.line([x, m + 6, x, int(H * 0.35)], fill=tuple(int(c * 0.6) for c in body), width=4)
    for sx in (int(W * 0.08), int(W * 0.92)):                           # latch clips
        d.rectangle([sx - 12, int(H * 0.4), sx + 12, int(H * 0.6)], fill=(70, 70, 74))
    if rng.random() < 0.6:
        d.ellipse([int(W * 0.7), int(H * 0.15), int(W * 0.82), int(H * 0.15) + int(W * 0.12)], fill=rng.choice([(220, 180, 20), (230, 120, 20), (30, 30, 30)]))
    if rng.random() < 0.6:
        d.rectangle([int(W * 0.2), int(H * 0.5), int(W * 0.8), int(H * 0.8)], fill=rng.choice([(235, 235, 235), (250, 210, 40)]))
        d.text((W // 2, int(H * 0.65)), rng.choice(["COOLANT", "FUSE", "MAX MIN", "OIL", "WASHER", "TOOLS"]), fill=(20, 20, 20),
               font=font(FONT_FILES[0], int(H * 0.16)), anchor="mm")
    return np.asarray(img), W / H


def engine_bay(rng):
    """Cluttered under-hood background."""
    base = SP.procedural_bg(rng) * rng.uniform(0.3, 0.8)
    if rng.random() < 0.5:
        nat = SP.natural_bg(rng) * rng.uniform(0.25, 0.6)
        base = base * 0.4 + nat * 0.6
    img = np.clip(base, 0, 255).astype(np.uint8).copy()
    for _ in range(rng.randint(5, 12)):
        c = rng.choice([(20, 20, 22), (45, 45, 50), (90, 92, 98), (170, 172, 178), (200, 150, 20), (30, 30, 30), (150, 30, 30)])
        kind = rng.random()
        x, y = rng.randint(-20, S), rng.randint(-20, S)
        if kind < 0.35:
            cv2.rectangle(img, (x, y), (x + rng.randint(20, 120), y + rng.randint(10, 80)), c, -1)
        elif kind < 0.55:
            cv2.circle(img, (x, y), rng.randint(6, 40), c, -1)
        else:
            pts = np.array([[x, y]] + [[x + rng.randint(-90, 90), y + rng.randint(-90, 90)] for _ in range(3)], np.int32)
            cv2.polylines(img, [pts], False, c, rng.randint(3, 12), cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.6)).astype(np.float32)


# ----------------------------------------------------------------------------------------------------- brake
def wheel_texture(rng, mode):
    """Face-on wheel/brake on transparent bg. mode: 'brake' (rotor+caliper visible), 'bare' (rotor, no wheel),
    'cover' (wheel with hub cap, brake NOT visible). -> RGB, alpha (both NxN), circle radius in px"""
    N = 600
    c = N // 2
    img = np.zeros((N, N, 3), np.uint8)
    alpha = np.zeros((N, N), np.uint8)
    R = int(N * 0.5) - 2

    def disc(r, col, a=True):
        cv2.circle(img, (c, c), int(r), col, -1, cv2.LINE_AA)
        if a:
            cv2.circle(alpha, (c, c), int(r), 255, -1, cv2.LINE_AA)

    rim = rng.choice(RIM_COL)
    metal = rng.choice([(150, 150, 156), (130, 128, 126), (165, 160, 150), (120, 100, 85)])   # steel / rusty
    if mode == "bare":
        R_rot = R
        disc(R_rot, metal)
    else:
        disc(R, (22, 22, 24))                                            # tyre
        for r in np.linspace(R * 0.74, R * 0.98, rng.randint(2, 5)):
            cv2.circle(img, (c, c), int(r), (36, 36, 40), 2, cv2.LINE_AA)
        disc(R * 0.72, rim)                                              # rim lip
        if mode == "cover":
            disc(R * 0.64, tuple(int(v * 0.92) for v in rim))
            for r in np.linspace(R * 0.18, R * 0.6, rng.randint(2, 5)):
                cv2.circle(img, (c, c), int(r), tuple(int(v * 0.65) for v in rim), 3, cv2.LINE_AA)
            for k in range(rng.choice([0, 5, 8, 12])):
                a = 2 * math.pi * k / 12
                cv2.line(img, (c, c), (int(c + R * 0.62 * math.cos(a)), int(c + R * 0.62 * math.sin(a))), tuple(int(v * 0.7) for v in rim), 3)
            disc(R * 0.14, tuple(int(v * 0.8) for v in rim))
            return img, alpha, R
        disc(R * 0.66, (28, 28, 32))                                     # barrel shadow
        R_rot = R * 0.58
        disc(R_rot, metal)
    # rotor: shading rings, slots or drilled holes
    for r in np.linspace(R_rot * 0.35, R_rot * 0.97, rng.randint(4, 9)):
        v = rng.randint(-14, 14)
        cv2.circle(img, (c, c), int(r), tuple(int(np.clip(x + v, 0, 255)) for x in metal), rng.randint(2, 6), cv2.LINE_AA)
    disc(R_rot * 0.34, tuple(int(v * 0.75) for v in metal), a=False)      # hat
    deco = rng.random()
    for k in range(rng.choice([36, 48, 60])):
        a = 2 * math.pi * k / 48
        if deco < 0.45:
            cv2.line(img, (int(c + R_rot * 0.45 * math.cos(a)), int(c + R_rot * 0.45 * math.sin(a))),
                     (int(c + R_rot * 0.95 * math.cos(a)), int(c + R_rot * 0.95 * math.sin(a))), (70, 70, 76), 3, cv2.LINE_AA)
        elif deco < 0.75:
            for rr in (0.55, 0.72, 0.88):
                cv2.circle(img, (int(c + R_rot * rr * math.cos(a + rr)), int(c + R_rot * rr * math.sin(a + rr))), 5, (30, 30, 34), -1)
    # caliper
    a0 = rng.uniform(0, 2 * math.pi)
    span = rng.uniform(0.7, 1.1)
    col = rng.choice(CAL_COL)
    outer, inner = R_rot * 1.0, R_rot * 0.52
    pts = [(c + outer * math.cos(a0 + span * t), c + outer * math.sin(a0 + span * t)) for t in np.linspace(0, 1, 12)]
    pts += [(c + inner * math.cos(a0 + span * t), c + inner * math.sin(a0 + span * t)) for t in np.linspace(1, 0, 12)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], col)
    cv2.polylines(img, [np.array(pts, np.int32)], True, tuple(int(v * 0.5) for v in col), 4, cv2.LINE_AA)
    mid = a0 + span / 2
    tx, ty = int(c + (R_rot * 0.78) * math.cos(mid)), int(c + (R_rot * 0.78) * math.sin(mid))
    cv2.putText(img, rng.choice(CALIPER_TXT), (tx - 40, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240) if sum(col) < 400 else (20, 20, 20), 2, cv2.LINE_AA)
    if mode == "brake":
        disc(R * 0.2, tuple(int(v * 0.8) for v in rim), a=False)            # hub cap
        for k in range(rng.choice([4, 5, 5, 6])):
            a = 2 * math.pi * k / 5 + 0.3
            cv2.circle(img, (int(c + R * 0.14 * math.cos(a)), int(c + R * 0.14 * math.sin(a))), 9, (200, 200, 205), -1, cv2.LINE_AA)
        ns = rng.choice([5, 5, 6, 7, 8, 10])                                 # spokes over the brake
        sw = rng.uniform(0.05, 0.11)
        for k in range(ns):
            a = 2 * math.pi * k / ns + rng.uniform(0, 0.4) * 0
            p0 = (c + R * 0.12 * math.cos(a), c + R * 0.12 * math.sin(a))
            p1 = (c + R * 0.7 * math.cos(a - sw), c + R * 0.7 * math.sin(a - sw))
            p2 = (c + R * 0.7 * math.cos(a + sw), c + R * 0.7 * math.sin(a + sw))
            cv2.fillConvexPoly(img, np.array([p0, p1, p2], np.int32), rim)
    else:
        disc(R * 0.14, (90, 90, 96), a=False)
        for k in range(5):
            a = 2 * math.pi * k / 5
            cv2.circle(img, (int(c + R * 0.2 * math.cos(a)), int(c + R * 0.2 * math.sin(a))), 12, (190, 190, 196), -1, cv2.LINE_AA)
    return img, alpha, R


def warp_alpha(bg, tex, alpha, quad):
    h, w = tex.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    wt = cv2.warpPerspective(tex, M, (S, S), flags=cv2.INTER_AREA)
    wa = cv2.warpPerspective(alpha, M, (S, S), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    return bg * (1 - wa) + wt.astype(np.float32) * wa, M


def wheel_quad(rng, size_frac, cx, cy, theta_deg):
    """Foreshortened square for a wheel seen at angle theta: width shrinks by cos, far side shrinks a little."""
    th = math.radians(theta_deg)
    a = size_frac * S / 2
    hw = a * math.cos(th)
    k = 0.14 * math.sin(th) * rng.choice([-1, 1])
    hl, hr = a * (1 + k), a * (1 - k)
    rot = math.radians(rng.gauss(0, 6))
    q = np.array([[-hw, -hl], [hw, -hr], [hw, hr], [-hw, hl]], np.float32)
    c, s = math.cos(rot), math.sin(rot)
    q = q @ np.array([[c, s], [-s, c]], np.float32)
    return q + np.array([cx, cy], np.float32)


def circle_poly(M, R, N=600, n=64):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([N / 2 + R * np.cos(t), N / 2 + R * np.sin(t)], 1).astype(np.float32)[None]
    return cv2.perspectiveTransform(pts, M)[0]


def shop_context(img, rng, quad):
    """Fender arch + body + floor around a wheel."""
    cx, cy = quad.mean(0)
    R = max(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])) / 2
    body = rng.choice([(235, 235, 235), (20, 20, 22), (170, 172, 178), (150, 25, 30), (30, 60, 130), (90, 95, 100)])
    cv2.rectangle(img, (int(cx - R * 2.6), int(cy - R * 2.0)), (int(cx + R * 2.6), int(cy + R * rng.uniform(0.2, 0.6))), body, -1)
    cv2.ellipse(img, (int(cx), int(cy - R * 0.1)), (int(R * 1.12), int(R * 1.12)), 0, 0, 360, (8, 8, 10), -1)
    fl = rng.choice([(110, 110, 115), (70, 72, 78), (150, 150, 150)])
    cv2.rectangle(img, (0, int(cy + R * 0.95)), (S, S), fl, -1)
    return img


# ----------------------------------------------------------------------------------------------------- common
def obj_quad(rng, aspect, wf, cx, cy):
    pw = wf * S
    ph = pw / aspect
    ang = float(np.clip(rng.gauss(0, 8), -30, 30)) * math.pi / 180
    c, s = math.cos(ang), math.sin(ang)
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]], np.float32)
    rot = base @ np.array([[c, s], [-s, c]], np.float32)
    jit = np.array([[rng.gauss(0, 0.06) * pw, rng.gauss(0, 0.06) * ph] for _ in range(4)], np.float32)
    return rot + jit + np.array([cx, cy], np.float32), ang


def _frames(img, rng, nprng, box, has):
    img = SP.camera_effects(img, rng)
    n = rng.choice([4, 6, 8, 12, 16, 16, 16])
    drift = np.array([rng.gauss(0, 2.0), rng.gauss(0, 2.0)])
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        sh = drift * t + nprng.normal(0, 0.4, 2)
        M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
        f = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
        f = f + nprng.normal(0, rng.uniform(0.5, 5), f.shape).astype(np.float32)
        if i == n - 1 and has:
            box = (box[0] + sh[0] / S, box[1] + sh[1] / S, box[2] + sh[0] / S, box[3] + sh[1] / S)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    q = rng.randint(55, 95)
    out = []
    for f in frames:
        ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1])
    return np.stack(out), box, n


def _place(rng, kind, half_w, half_h, wf):
    if kind == "framed":
        cx = rng.uniform(half_w + 0.05, 1 - half_w - 0.05) if half_w + 0.05 < 0.5 else 0.5
        cy = rng.uniform(half_h + 0.05, 1 - half_h - 0.05) if half_h + 0.05 < 0.5 else 0.5
    else:
        cx, cy = rng.uniform(-0.15, 1.15), rng.uniform(-0.15, 1.15)
    return cx, cy


def _size(rng, kind, lo, hi):
    if kind == "framed":
        return rng.uniform(lo, hi)
    if kind == "partial":
        return math.exp(rng.uniform(math.log(0.25), math.log(1.5)))
    return rng.choice([rng.uniform(0.06, 0.20), rng.uniform(0.95, 1.6)])


def _tail(box_quad_pts, kind_ok_fn):
    pass


def make_battery(seed):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    kind = rng.choices(["framed", "partial", "far_near", "covered", "negative"], [0.25, 0.26, 0.07, 0.20, 0.22])[0]
    bg = engine_bay(rng) if rng.random() < 0.8 else (SP.natural_bg(rng) if rng.random() < 0.6 else SP.procedural_bg(rng))
    present = done = covered = 0
    box = (0.0, 0.0, 0.0, 0.0)
    quad = None
    if kind in ("framed", "partial", "far_near", "covered"):
        if kind == "covered":
            tex, aspect = cover_texture(rng)
            terms = []
            pk = "framed" if rng.random() < 0.65 else "partial"
            wf = _size(rng, pk, 0.32, 0.85)
        else:
            tex, aspect, terms = battery_texture(rng)
            pk = kind
            wf = _size(rng, kind, 0.32, 0.85)
        cx, cy = _place(rng, pk, wf / 2, wf / aspect / 2, wf)
        quad, ang = obj_quad(rng, aspect, wf, cx * S, cy * S)
        img = bg.copy()
        if rng.random() < 0.25:                                           # thin tray lip only (a bare battery has no box around it)
            c = quad.mean(0)
            cv2.fillConvexPoly(img, ((quad - c) * rng.uniform(1.02, 1.07) + c).astype(np.int32), (14, 14, 16))
        img, M = _warp_rgb(img, tex, quad)
        if terms:
            pts = cv2.perspectiveTransform(np.array([[[x, y] for x, y, _ in terms]], np.float32), M)[0]
            for (px, py), (_, _, pos) in zip(pts, terms):                 # cables leaving the terminals
                a = rng.uniform(-math.pi, 0) if rng.random() < 0.7 else rng.uniform(0, math.pi)
                ln = rng.uniform(0.08, 0.22) * S
                cv2.line(img, (int(px), int(py)), (int(px + ln * math.cos(a)), int(py + ln * math.sin(a))),
                         (190, 25, 25) if pos else (22, 22, 24), rng.randint(3, 7), cv2.LINE_AA)
        img = img.astype(np.float32)
    else:
        sub = rng.choices(["bay", "container", "plate", "wheel", "scene"], [0.2, 0.3, 0.15, 0.15, 0.2])[0]
        img = bg.copy()
        wf = rng.uniform(0.3, 0.85)
        cx, cy = rng.uniform(0.2, 0.8) * S, rng.uniform(0.2, 0.8) * S
        if sub == "container":
            tex, aspect = container_texture(rng)
            q, _ = obj_quad(rng, aspect, wf, cx, cy)
            c = q.mean(0)
            if rng.random() < 0.6:
                cv2.fillConvexPoly(img, ((q - c) * 1.15 + c).astype(np.int32), (14, 14, 16))
            img, _ = _warp_rgb(img, tex, q)
        elif sub == "plate":
            tex, aspect = SP.plate_texture(rng)
            q, _ = obj_quad(rng, aspect, wf, cx, cy)
            img, _ = _warp_rgb(img, tex, q)
        elif sub == "wheel":
            tex, alpha, _ = wheel_texture(rng, rng.choice(["brake", "cover", "bare"]))
            img, _ = warp_alpha(img, tex, alpha, wheel_quad(rng, wf, cx, cy, rng.uniform(0, 35)))
        img = img.astype(np.float32)
    if quad is not None:
        vis = SP.visible_fraction(quad)
        x0, y0 = quad.min(0) / S
        x1, y1 = quad.max(0) / S
        wfrac = float(max(x1 - x0, y1 - y0))
        box = (float(x0), float(y0), float(x1), float(y1))
        seen = int(vis >= 0.20 and wfrac >= 0.12)
        if kind == "covered":
            covered = seen              # a covered battery is NOT a valid target: present stays 0
        else:
            present = seen
        top = np.linalg.norm(quad[1] - quad[0]); bot = np.linalg.norm(quad[2] - quad[3])
        lef = np.linalg.norm(quad[3] - quad[0]); rig = np.linalg.norm(quad[2] - quad[1])
        inside = quad.min() >= 6 and quad.max() <= S - 6
        done = int(present and inside and 0.30 <= wfrac <= 0.88 and abs(ang) <= math.radians(22)
                   and 0.7 <= top / bot <= 1.4 and 0.7 <= lef / rig <= 1.4)
    frames, box, n = _frames(img, rng, nprng, box, quad is not None and kind != "covered")
    return frames, dict(present=present, done=done, box=box if present else (0.0, 0.0, 0.0, 0.0), kind=kind, n=n, covered=covered,
                        cbox=box if covered else (0.0, 0.0, 0.0, 0.0))


def _warp_rgb(bg, tex, quad):
    h, w = tex.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(tex, M, (S, S), flags=cv2.INTER_AREA if w > 3 * S else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (S, S), flags=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(mask, (0, 0), 0.6).astype(np.float32)[..., None] / 255.0
    return bg * (1 - m) + warped.astype(np.float32) * m, M


def make_brake(seed):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    kind = rng.choices(["framed", "partial", "far_near", "negative"], [0.30, 0.32, 0.08, 0.30])[0]
    bg = SP.natural_bg(rng) if rng.random() < 0.5 else SP.procedural_bg(rng)
    if rng.random() < 0.3:
        bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(1, 4))
    present = done = 0
    box = (0.0, 0.0, 0.0, 0.0)
    poly = None
    theta = 0.0
    if kind != "negative":
        mode = rng.choices(["brake", "bare"], [0.78, 0.22])[0]
        tex, alpha, R = wheel_texture(rng, mode)
        wf = _size(rng, kind, 0.38, 0.9)
        theta = float(np.clip(abs(rng.gauss(0, 22)), 0, 55))
        cx, cy = _place(rng, kind, wf / 2, wf / 2, wf)
        quad = wheel_quad(rng, wf * 600 / (2 * R), cx * S, cy * S, theta)
        img = bg.copy()
        if rng.random() < 0.65 and mode == "brake":
            img = shop_context(img, rng, quad)
        img, M = warp_alpha(img, tex, alpha, quad)
        poly = circle_poly(M, R)
    else:
        sub = rng.choices(["cover", "scene", "round", "battery", "plate", "tyre"], [0.28, 0.18, 0.14, 0.14, 0.12, 0.14])[0]
        img = bg.copy()
        wf = rng.uniform(0.3, 0.9)
        cx, cy = rng.uniform(0.2, 0.8) * S, rng.uniform(0.2, 0.8) * S
        if sub == "cover":
            tex, alpha, _ = wheel_texture(rng, "cover")
            q = wheel_quad(rng, wf, cx, cy, rng.uniform(0, 45))
            if rng.random() < 0.6:
                img = shop_context(img, rng, q)
            img, _ = warp_alpha(img, tex, alpha, q)
        elif sub == "tyre":                                       # tyre sidewall only: black disc with rings
            N = 600
            tex = np.full((N, N, 3), 22, np.uint8)
            alpha = np.zeros((N, N), np.uint8)
            cv2.circle(alpha, (N // 2, N // 2), N // 2 - 2, 255, -1)
            for r in range(60, 290, rng.randint(14, 40)):
                cv2.circle(tex, (N // 2, N // 2), r, (40, 40, 44), 3)
            img, _ = warp_alpha(img, tex, alpha, wheel_quad(rng, wf, cx, cy, rng.uniform(0, 45)))
        elif sub == "round":                                       # clocks, discs, buckets: round but not a brake
            im = SP.real_images()[rng.randrange(len(SP.real_images()))]
            tex = cv2.resize(im, (600, 600))
            alpha = np.zeros((600, 600), np.uint8)
            cv2.circle(alpha, (300, 300), 296, 255, -1)
            img, _ = warp_alpha(img, tex, alpha, wheel_quad(rng, wf, cx, cy, rng.uniform(0, 45)))
        elif sub == "battery":
            tex, aspect, _ = battery_texture(rng)
            q, _ = obj_quad(rng, aspect, wf, cx, cy)
            img, _ = _warp_rgb(img, tex, q)
        elif sub == "plate":
            tex, aspect = SP.plate_texture(rng)
            q, _ = obj_quad(rng, aspect, wf, cx, cy)
            img, _ = _warp_rgb(img, tex, q)
    img = img.astype(np.float32)
    if poly is not None:
        frame = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
        area, _ = cv2.intersectConvexConvex(poly.astype(np.float32), frame)
        vis = float(area) / max(cv2.contourArea(poly.astype(np.float32)), 1e-6)
        x0, y0 = poly.min(0) / S
        x1, y1 = poly.max(0) / S
        wfrac = float(max(x1 - x0, y1 - y0))
        box = (float(x0), float(y0), float(x1), float(y1))
        present = int(vis >= 0.25 and wfrac >= 0.12)
        inside = poly.min() >= 6 and poly.max() <= S - 6
        done = int(present and inside and 0.38 <= wfrac <= 0.92 and theta <= 40)
    frames, box, n = _frames(img, rng, nprng, box, poly is not None)
    return frames, dict(present=present, done=done, box=box, kind=kind, n=n)


def make_clip(task, seed):
    return make_battery(seed) if task == "battery" else make_brake(seed)
