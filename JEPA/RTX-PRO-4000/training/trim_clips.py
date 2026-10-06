"""Cut 18 real sub-clips (10s,20s,...,180s) out of the existing test_3min_1080p.mp4,
preserving real encoded video (not synthetic tensors) so the HTTP client can send
genuine video files to the Triton server."""
import cv2
import os

SRC = "test_3min_1080p.mp4"
OUT_DIR = "clips"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"source: {fps} fps, {w}x{h}")

durations = list(range(10, 181, 10))
for dur_s in durations:
    n_frames_needed = round(dur_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_path = os.path.join(OUT_DIR, f"clip_{dur_s}s.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < n_frames_needed:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame)
        n += 1
    vw.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{out_path}: {n} frames, {size_mb:.1f} MB")

cap.release()
print("done")
