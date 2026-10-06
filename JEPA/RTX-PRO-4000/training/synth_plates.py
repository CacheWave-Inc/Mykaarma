"""Synthetic license-plate scenes for training a guidance head on frozen V-JEPA 2 features.

make_clip(seed) -> (frames uint8 [n,256,256,3] RGB, labels dict)
labels: present (1 if a plate is meaningfully visible), done (1 if complete + well framed),
        box = (x0,y0,x1,y1) of the FULL plate in [0,1] frame coords (may lie outside [0,1]).
"""
import functools
import math
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 256
FONT_FILES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Bold.otf",
]
STATES = ["CALIFORNIA", "TEXAS", "FLORIDA", "NEW YORK", "OHIO", "NEVADA", "UTAH", "ARIZONA", "OREGON",
          "WASHINGTON", "COLORADO", "GEORGIA", "VIRGINIA", "ILLINOIS", "MICHIGAN", "KANSAS", "IDAHO", "MAINE"]
SLOGANS = ["THE GOLDEN STATE", "LONE STAR", "SUNSHINE STATE", "EXCELSIOR", "BEYOND IMAGINATION", "GRAND CANYON STATE",
           "LIVE FREE OR DIE", "DISCOVER", "EVERGREEN STATE", "CENTENNIAL STATE", "GARDEN STATE"]
WORDS = ["EXIT", "STOP", "SALE", "OPEN", "PARKING", "SPEED LIMIT", "NO ENTRY", "SERVICE", "AUTO REPAIR", "WELCOME",
         "DEALER", "FOR RENT", "CAUTION", "ONE WAY", "TOWING", "PRICE 49", "NEXT 2 MI", "GAS", "ROUTE 66", "MENU"]
PLATE_BG = [(250, 250, 248), (244, 240, 225), (255, 224, 80), (235, 240, 250), (250, 200, 120), (225, 235, 225),
            (255, 255, 255), (240, 232, 200), (30, 30, 34), (20, 60, 140)]


@functools.lru_cache(maxsize=256)
def font(path, size):
    return ImageFont.truetype(path, size)


@functools.lru_cache(maxsize=1)
def real_images():
    from skimage import data
    out = []
    for n in ["astronaut", "brick", "camera", "cat", "chelsea", "coffee", "coins", "grass", "gravel", "horse",
              "moon", "rocket", "clock", "page", "text"]:
        im = getattr(data, n)()
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        out.append(im[..., :3].astype(np.uint8))
    return out


def rand_plate_text(rng):
    L = string.ascii_uppercase.replace("I", "").replace("O", "")
    D = string.digits
    f = rng.choice(["LLL DDDD", "DLLL DDD", "LLL-DDD", "LL-DDDD", "DDD LLL", "LLLL DDD", "LLDDLLL", "DLLLDDD", "LL DD LLL"])
    return "".join(rng.choice(L) if c == "L" else rng.choice(D) if c == "D" else c for c in f)


def plate_texture(rng):
    """Flat plate image + its aspect (w/h)."""
    kind = rng.random()
    if kind < 0.75:
        W, H = 600, 300
    elif kind < 0.9:
        W, H = 1040, 220
    else:
        aspect = rng.uniform(1.4, 3.2)
        W, H = int(300 * aspect), 300
    bg = rng.choice(PLATE_BG)
    dark = sum(bg) < 300
    fg = (235, 235, 235) if dark else rng.choice([(20, 20, 25), (10, 30, 100), (120, 20, 20), (15, 70, 35)])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    eu = W / H > 4 and rng.random() < 0.7
    x_off = 0
    if eu:
        d.rectangle([0, 0, int(H * 0.28), H], fill=(0, 50, 160))
        d.text((int(H * 0.14), H // 2), "EU", fill=(255, 220, 0), font=font(FONT_FILES[1], int(H * 0.28)), anchor="mm")
        x_off = int(H * 0.28)
    if rng.random() < 0.9:
        bw = rng.randint(4, 10)
        d.rectangle([bw, bw, W - bw, H - bw], outline=fg, width=rng.randint(2, 6))
    ff = rng.choice(FONT_FILES)
    two_line = rng.random() < 0.06 and not eu
    txt = rand_plate_text(rng)
    th = int(H * (0.62 if W / H > 3 else 0.46))
    if W / H < 3 and rng.random() < 0.8:
        d.text((W // 2, int(H * 0.15)), rng.choice(STATES), fill=fg, font=font(ff, int(H * 0.13)), anchor="mm")
        if rng.random() < 0.6:
            d.text((W // 2, int(H * 0.88)), rng.choice(SLOGANS), fill=fg, font=font(ff, int(H * 0.09)), anchor="mm")
    if two_line:
        a, b = txt[: len(txt) // 2], txt[len(txt) // 2:]
        d.text((W // 2, int(H * 0.36)), a, fill=fg, font=font(ff, int(H * 0.30)), anchor="mm")
        d.text((W // 2, int(H * 0.68)), b, fill=fg, font=font(ff, int(H * 0.30)), anchor="mm")
    else:
        f = font(ff, th)
        while d.textlength(txt, font=f) > (W - x_off) * 0.9 and th > 20:
            th -= 4
            f = font(ff, th)
        d.text((x_off + (W - x_off) // 2, int(H * 0.52)), txt, fill=fg, font=f, anchor="mm")
    if rng.random() < 0.4:
        for sx in (int(W * 0.1), int(W * 0.9)):
            d.ellipse([sx - 7, int(H * 0.12) - 7, sx + 7, int(H * 0.12) + 7], fill=(150, 150, 150))
    if rng.random() < 0.3:
        sx, sy = int(W * 0.9), int(H * 0.25)
        d.rectangle([sx - 28, sy - 20, sx + 28, sy + 20], fill=rng.choice([(200, 30, 30), (30, 90, 200), (240, 200, 30)]))
    arr = np.asarray(img).astype(np.float32)
    arr *= rng.uniform(0.9, 1.0)
    arr += rng.uniform(-8, 8)
    return np.clip(arr, 0, 255).astype(np.uint8), W / H


def sign_texture(rng):
    W, H = rng.choice([(600, 300), (500, 360), (700, 240), (400, 400), (600, 200)])
    bg, fg = rng.choice([((20, 110, 60), (245, 245, 245)), ((200, 30, 30), (255, 255, 255)), ((255, 255, 255), (20, 20, 20)),
                         ((250, 210, 40), (20, 20, 20)), ((20, 70, 160), (255, 255, 255)), ((40, 40, 40), (240, 240, 100))])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, W - 6, H - 6], outline=fg, width=6)
    ff = rng.choice(FONT_FILES)
    lines = [rng.choice(WORDS)] + ([rng.choice(WORDS)] if rng.random() < 0.5 else [])
    hh = H // (len(lines) + 1)
    for i, t in enumerate(lines):
        sz = int(hh * 0.8)
        f = font(ff, sz)
        while d.textlength(t, font=f) > W * 0.88 and sz > 16:
            sz -= 4
            f = font(ff, sz)
        d.text((W // 2, int(H * (i + 1) / (len(lines) + 1))), t, fill=fg, font=f, anchor="mm")
    return np.asarray(img), W / H


def blank_plate_texture(rng):
    """Plate-shaped rectangle with NO characters (empty recess, frame, barcode bars)."""
    W, H = 600, 300
    img = Image.new("RGB", (W, H), rng.choice([(245, 245, 245), (40, 40, 44), (200, 200, 205)]))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, W - 8, H - 8], outline=(60, 60, 60), width=6)
    if rng.random() < 0.5:
        x = 80
        while x < W - 80:
            w = rng.randint(3, 14)
            d.rectangle([x, 90, x + w, 210], fill=(20, 20, 20))
            x += w + rng.randint(4, 12)
    elif rng.random() < 0.5:
        d.text((W // 2, int(H * 0.88)), rng.choice(["DEALER NAME", "MOTORS", "AUTO SALES", "CITY FORD"]),
               fill=(230, 230, 230), font=font(FONT_FILES[0], 44), anchor="mm")
    return np.asarray(img), W / H


def natural_bg(rng):
    imgs = real_images()
    im = imgs[rng.randrange(len(imgs))]
    h, w = im.shape[:2]
    s = int(min(h, w) * rng.uniform(0.35, 1.0))
    y, x = rng.randint(0, h - s), rng.randint(0, w - s)
    crop = cv2.resize(im[y:y + s, x:x + s], (S, S), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    return crop.astype(np.float32)


def procedural_bg(rng):
    base = np.full((S, S, 3), [rng.uniform(30, 200)] * 3, np.float32) + np.array([rng.uniform(-25, 25) for _ in range(3)], np.float32)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(5, 30), (S, S, 1)).astype(np.float32)
    base += cv2.GaussianBlur(noise[..., 0], (0, 0), rng.uniform(0.5, 6))[..., None]
    gx = np.linspace(-1, 1, S, dtype=np.float32)
    base += (gx[None, :, None] * rng.uniform(-30, 30) + gx[:, None, None] * rng.uniform(-30, 30))
    return np.clip(base, 0, 255)


def car_context(img, rng, quad, empty_recess=False):
    """Draw a car-rear-like body around the plate location (in-place, float image)."""
    cx, cy = quad.mean(0)
    pw = np.ptp(quad[:, 0]) + 1
    ph = np.ptp(quad[:, 1]) + 1
    body = rng.choice([(235, 235, 235), (20, 20, 22), (170, 172, 178), (150, 25, 30), (30, 60, 130), (90, 95, 100)])
    w, h = pw * rng.uniform(2.2, 4.5), ph * rng.uniform(2.5, 5.5)
    x0, y0, x1, y1 = int(cx - w / 2), int(cy - h * rng.uniform(0.35, 0.65)), int(cx + w / 2), int(cy + h * rng.uniform(0.35, 0.65))
    cv2.rectangle(img, (x0, y0), (x1, y1), body, -1)
    shade = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.75, 1.0), S, dtype=np.float32)[:, None, None]
    img *= np.where(((np.arange(S)[None, :] >= x0) & (np.arange(S)[None, :] <= x1) & (np.arange(S)[:, None] >= y0) & (np.arange(S)[:, None] <= y1))[..., None], shade, 1.0)
    by0 = int(y1 - (y1 - y0) * rng.uniform(0.2, 0.35))
    cv2.rectangle(img, (x0, by0), (x1, y1), tuple(int(c * 0.6) for c in body), -1)
    for sx in (x0, x1):
        lx0, lx1 = (sx - int(w * 0.04), sx + int(w * 0.12)) if sx == x0 else (sx - int(w * 0.12), sx + int(w * 0.04))
        cv2.rectangle(img, (lx0, int(cy - ph * 0.9)), (lx1, int(cy - ph * 0.1)), (170, 20, 25), -1)
    cv2.line(img, (x0, int(cy - ph * 0.85)), (x1, int(cy - ph * 0.85)), tuple(int(c * 0.5) for c in body), 2)
    r = quad.astype(np.int32)
    cv2.fillConvexPoly(img, cv2.convexHull(r), (40, 40, 44) if empty_recess else (25, 25, 28))
    return img


def random_quad(rng, aspect, width_frac, cx, cy):
    pw = width_frac * S
    ph = pw / aspect
    ang = float(np.clip(rng.gauss(0, 5), -22, 22)) * math.pi / 180
    c, s = math.cos(ang), math.sin(ang)
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]], np.float32)
    rot = base @ np.array([[c, s], [-s, c]], np.float32)
    jit = np.array([[rng.gauss(0, 0.035) * pw, rng.gauss(0, 0.035) * ph] for _ in range(4)], np.float32)
    return rot + jit + np.array([cx, cy], np.float32), ang


def warp_onto(bg, tex, quad):
    h, w = tex.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(tex, M, (S, S), flags=cv2.INTER_AREA if w > 3 * S else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (S, S), flags=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(mask, (0, 0), 0.6).astype(np.float32)[..., None] / 255.0
    return bg * (1 - m) + warped.astype(np.float32) * m


def visible_fraction(quad):
    frame = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    area, _ = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    return float(area) / max(cv2.contourArea(quad.astype(np.float32)), 1e-6)


def camera_effects(img, rng):
    if rng.random() < 0.35:
        k = rng.choice([3, 5, 7, 9])
        kern = np.zeros((k, k), np.float32)
        a = rng.uniform(0, math.pi)
        cv2.line(kern, (k // 2, k // 2), (int(k // 2 + math.cos(a) * k // 2), int(k // 2 + math.sin(a) * k // 2)), 1, 1)
        kern /= max(kern.sum(), 1)
        img = cv2.filter2D(img, -1, kern)
    elif rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.8))
    img = img * rng.uniform(0.55, 1.25) + rng.uniform(-25, 25)
    img = img * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    return np.clip(img, 0, 255)


def make_clip(seed, force=None):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    kind = force or rng.choices(["framed", "partial", "far_near", "negative"], [0.30, 0.32, 0.08, 0.30])[0]
    bg = natural_bg(rng) if rng.random() < 0.55 else procedural_bg(rng)
    if rng.random() < 0.3:
        bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(1, 4))

    quad = None
    present = done = 0
    box = (0.0, 0.0, 0.0, 0.0)
    ang = 0.0
    if kind != "negative":
        tex, aspect = plate_texture(rng)
        if kind == "framed":
            wf = rng.uniform(0.30, 0.85)
        elif kind == "partial":
            wf = math.exp(rng.uniform(math.log(0.25), math.log(1.5)))
        else:
            wf = rng.choice([rng.uniform(0.06, 0.20), rng.uniform(0.9, 1.6)])
        half_w = wf / 2
        half_h = wf / aspect / 2
        if kind == "framed":
            cx = rng.uniform(half_w + 0.05, 1 - half_w - 0.05) if half_w + 0.05 < 0.5 else 0.5
            cy = rng.uniform(half_h + 0.05, 1 - half_h - 0.05) if half_h + 0.05 < 0.5 else 0.5
        else:
            cx, cy = rng.uniform(-0.15, 1.15), rng.uniform(-0.15, 1.15)
        quad, ang = random_quad(rng, aspect, wf, cx * S, cy * S)
        if rng.random() < 0.6:
            bg = car_context(bg.copy(), rng, quad)
        img = warp_onto(bg, tex, quad)
    else:
        sub = rng.choices(["scene", "car_empty", "sign", "blank_plate", "text"], [0.25, 0.2, 0.25, 0.15, 0.15])[0]
        cx, cy = rng.uniform(0.2, 0.8) * S, rng.uniform(0.2, 0.8) * S
        wf = rng.uniform(0.25, 0.9)
        img = bg
        if sub in ("sign", "blank_plate", "car_empty", "text"):
            if sub == "text":
                tex = real_images()[-rng.randint(1, 2)]
                tex = cv2.resize(tex, (600, 300))
                aspect = 2.0
            elif sub == "sign":
                tex, aspect = sign_texture(rng)
            else:
                tex, aspect = blank_plate_texture(rng)
            q, _ = random_quad(rng, aspect, wf, cx, cy)
            if sub in ("car_empty", "blank_plate") or rng.random() < 0.4:
                img = car_context(img.copy(), rng, q, empty_recess=True)
            img = warp_onto(img, tex, q) if sub != "car_empty" else img
        elif rng.random() < 0.4:
            q, _ = random_quad(rng, 2.0, wf, cx, cy)
            img = car_context(img.copy(), rng, q)

    if quad is not None:
        vis = visible_fraction(quad)
        x0, y0 = quad.min(0) / S
        x1, y1 = quad.max(0) / S
        wfrac = float(x1 - x0)
        box = (float(x0), float(y0), float(x1), float(y1))
        present = int(vis >= 0.20 and wfrac >= 0.12)
        top = np.linalg.norm(quad[1] - quad[0])
        bot = np.linalg.norm(quad[2] - quad[3])
        lef = np.linalg.norm(quad[3] - quad[0])
        rig = np.linalg.norm(quad[2] - quad[1])
        inside = quad.min() >= 6 and quad.max() <= S - 6
        done = int(present and inside and 0.28 <= wfrac <= 0.88 and abs(ang) <= math.radians(13)
                   and 0.8 <= top / bot <= 1.25 and 0.8 <= lef / rig <= 1.25)

    img = camera_effects(img, rng)
    n = rng.choice([4, 6, 8, 12, 16, 16, 16])
    drift = np.array([rng.gauss(0, 2.0), rng.gauss(0, 2.0)])
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        sh = drift * t + nprng.normal(0, 0.4, 2)
        M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
        f = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
        f = f + nprng.normal(0, rng.uniform(0.5, 5), f.shape).astype(np.float32)
        if i == n - 1 and quad is not None:
            box = (box[0] + sh[0] / S, box[1] + sh[1] / S, box[2] + sh[0] / S, box[3] + sh[1] / S)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    q = rng.randint(55, 95)
    out = []
    for f in frames:
        ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1])
    labels = dict(present=present, done=done, box=box, kind=kind, n=n)
    return np.stack(out), labels
