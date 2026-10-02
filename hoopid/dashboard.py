"""Coach dashboard video: original frame on the left, live per-player panel on the right."""
from __future__ import annotations

import csv
import os
import subprocess
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .video import iter_frames

PANEL_W = 640
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "C:/Windows/Fonts/arial.ttf"]


def _font(size, bold=False):
    paths = ([p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf") for p in FONT_PATHS] if bold else []) + FONT_PATHS
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (170, 80, 220), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
           (0, 128, 128), (220, 190, 255)]


def pid_color(pid: str) -> tuple:
    """Distinct RGB colour per player number (fixed palette)."""
    return PALETTE[(int(pid.split("_")[1]) - 1) % len(PALETTE)]


def _rolling_max_speed(speed: np.ndarray, win: int = 30) -> np.ndarray:
    s = np.where(np.isnan(speed), -1, speed)
    med = np.full_like(s, -1.0)
    for i in range(len(s)):
        w = s[max(0, i - win + 1):i + 1]
        w = w[w >= 0]
        if len(w) >= win // 3:
            med[i] = np.median(w)
    return np.maximum.accumulate(med)


def render(video: str, tracks_csv: str, timeline: dict, summary: dict, meta: dict, out_path: str,
           ball_npz: str | None = None):
    rows = defaultdict(list)
    for r in csv.DictReader(open(tracks_csv)):
        if r["observed"] == "1":
            rows[int(r["frame"])].append(r)
    balls = defaultdict(list)
    if ball_npz and os.path.exists(ball_npz):
        z = np.load(ball_npz)
        for f, b, s in zip(z["frame"], z["box"], z["score"]):
            if s >= meta["config"]["ball_min_score"]:
                balls[int(f)].append(b)
    pids = sorted(summary)
    vmax = {p: _rolling_max_speed(timeline[p]["speed"]) for p in pids}
    # minimap bounds from all estimated positions
    allpos = np.array([xy for p in pids for xy in timeline[p]["pos"].values()]) if pids else np.zeros((1, 2))
    xlo, xhi = np.percentile(allpos[:, 0], [1, 99]); zlo, zhi = np.percentile(allpos[:, 1], [1, 99])
    xlo, xhi = xlo - 1, xhi + 1; zlo, zhi = max(0, zlo - 1), zhi + 1
    f_title, f_hdr, f_row, f_small = _font(22, True), _font(14, True), _font(16), _font(12)
    tmp = out_path + ".tmp.mp4"
    vw = None
    total_t = None
    for fr in iter_frames(video):
        img = fr.image.copy()
        H, W = img.shape[:2]
        # ---- left: video with labels
        for r in rows.get(fr.idx, []):
            x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
            if r["status"] == "CONFIRMED":
                col = pid_color(r["player_id"])[::-1]
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                lab = r["player_id"].replace("PLAYER_", "P")
                cv2.putText(img, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(img, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            elif r["status"] == "UNCERTAIN":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 1)
                cv2.putText(img, "?", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        for b in balls.get(fr.idx, []):
            c = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
            cv2.circle(img, c, max(4, int((b[2] - b[0]) / 2) + 2), (0, 140, 255), 2)
        # ---- right: panel
        panel = Image.new("RGB", (PANEL_W, H), (24, 26, 31))
        d = ImageDraw.Draw(panel)
        d.text((16, 12), "Coach panel", font=f_title, fill=(255, 255, 255))
        d.text((16, 42), f"time {fr.t:5.1f} s   frame {fr.idx}", font=f_small, fill=(170, 175, 185))
        cols = [("Player", 16), ("Status", 74), ("On screen", 176), ("With ball", 270), ("Touches", 362),
                ("Dist.", 446), ("Run", 510), ("Speed", 568)]
        units = ["", "", "s", "s", "", "m", "m", "m/s"]
        y0 = 68
        for (name, x), u in zip(cols, units):
            d.text((x, y0), name, font=f_hdr, fill=(150, 160, 175))
            d.text((x, y0 + 17), u, font=f_small, fill=(120, 128, 140))
        d.line((12, y0 + 36, PANEL_W - 12, y0 + 36), fill=(60, 64, 72))
        y = y0 + 42
        for p in pids:
            tl = timeline[p]
            f = fr.idx
            on = f in tl["frames"]
            with_ball = f in tl["ball_frames"]
            touches = sum(1 for s in tl["touch_starts"] if s <= f)
            sp = tl["speed"][f] if f < len(tl["speed"]) else np.nan
            colr = pid_color(p)
            d.rectangle((16, y + 3, 28, y + 15), fill=colr)
            d.text((34, y), p.replace("PLAYER_", "P"), font=f_row, fill=(255, 255, 255))
            st, stc = ("with ball", (255, 170, 60)) if with_ball else (("on screen", (120, 220, 120)) if on else ("—", (110, 115, 125)))
            d.text((cols[1][1], y), st, font=f_row, fill=stc)
            vals = [f"{tl['vis'][f]:.1f}", f"{tl['ball'][f]:.1f}", f"{touches}", f"{tl['dist'][f]:.0f}",
                    f"{tl['run'][f]:.0f}", f"{sp:.1f}" if on and not np.isnan(sp) else "—"]
            for (name, x), v in zip(cols[2:], vals):
                d.text((x, y), v, font=f_row, fill=(230, 232, 236))
            y += 26
        # ---- minimap (top-down estimate)
        my0, my1 = y + 18, H - 92
        mx0, mx1 = 16, PANEL_W - 16
        d.text((16, y + 2), "Mini-map (top-down, estimated; bottom = closer to camera)", font=f_small, fill=(150, 160, 175))
        d.rectangle((mx0, my0, mx1, my1), outline=(70, 74, 82))

        def to_map(x, z):
            px = mx0 + (x - xlo) / (xhi - xlo) * (mx1 - mx0)
            py = my1 - (z - zlo) / (zhi - zlo) * (my1 - my0)
            return px, py
        for p in pids:
            pos = timeline[p]["pos"]
            trail = [pos[k] for k in range(max(0, fr.idx - 60), fr.idx + 1) if k in pos]
            if len(trail) >= 2:
                d.line([to_map(*q) for q in trail], fill=pid_color(p), width=2)
            if fr.idx in pos:
                cx, cy = to_map(*pos[fr.idx])
                d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=pid_color(p))
                d.text((cx + 8, cy - 8), p.replace("PLAYER_", "P"), font=f_small, fill=(255, 255, 255))
        notes = ["P01–P09 are registration IDs (no jersey numbers in this video).",
                 "Only frames with a confirmed ID are counted; UNKNOWN is excluded.",
                 f"Distance/speed are estimates without court calibration: height {meta['assumptions']['player_height_m']} m,",
                 f"field of view {meta['assumptions']['fov_long_side_deg']}°; expected error ±30%.",
                 "'With ball' = ball in hands / dribbling (warm-up: many have their own ball)."]
        for k, t in enumerate(notes):
            d.text((16, H - 86 + 16 * k), t, font=f_small, fill=(140, 146, 158))
        frame = np.hstack([img, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)])
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-pix_fmt", "yuv420p", out_path], check=True)
    os.remove(tmp)
