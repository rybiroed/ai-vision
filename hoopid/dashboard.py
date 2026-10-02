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


MONO_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "C:/Windows/Fonts/consolab.ttf"]
SS = 2  # panel is drawn at 2x and downscaled for smooth edges

BG = (17, 20, 26)
CARD = (27, 32, 41)
CARD_ON = (37, 44, 56)
INK = (240, 236, 228)
MUTED = (138, 148, 163)
FAINT = (78, 86, 99)
ORANGE = (240, 138, 36)
GREEN = (86, 196, 118)


def _mono(size):
    for p in MONO_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return _font(size, True)


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


class Panel:
    """Coach panel renderer (all sizes below are in final pixels; drawn at SS x)."""

    def __init__(self, height: int, pids: list, summary: dict, timeline: dict, meta: dict, duration: float,
                 bounds: tuple):
        self.H, self.W = height, PANEL_W
        self.pids, self.summary, self.tl, self.meta = pids, summary, timeline, meta
        self.duration = duration
        self.xlo, self.xhi, self.zlo, self.zhi = bounds
        f = lambda n, b=False: _font(n * SS, b)
        self.f_eyebrow, self.f_title, self.f_clock = f(12, True), f(24, True), _mono(26 * SS)
        self.f_small, self.f_hdr, self.f_row, self.f_rowb = f(12), f(11, True), f(15), f(15, True)
        self.f_num, self.f_kpi, self.f_pill, self.f_note = _mono(15 * SS), _mono(22 * SS), f(10, True), f(11)
        self.max_ball = max([summary[p]["ball_s"] for p in pids] + [1.0])
        self.max_dist = max([summary[p]["dist_m"] for p in pids] + [1.0])

    # helpers in final-pixel units
    def _r(self, d, box, radius, fill=None, outline=None, width=1):
        d.rounded_rectangle([v * SS for v in box], radius=radius * SS, fill=fill, outline=outline, width=width * SS)

    def _t(self, d, xy, text, font, fill, anchor="la"):
        d.text((xy[0] * SS, xy[1] * SS), text, font=font, fill=fill, anchor=anchor)

    def draw(self, frame_idx: int, t: float, n_balls: int, n_unknown: int) -> np.ndarray:
        W, H = self.W, self.H
        img = Image.new("RGB", (W * SS, H * SS), BG)
        d = ImageDraw.Draw(img)
        # header
        self._t(d, (20, 18), "COACH PANEL  ·  HOOPID", self.f_eyebrow, ORANGE)
        self._t(d, (20, 36), "Player tracking", self.f_title, INK)
        self._t(d, (W - 20, 20), _clock(t), self.f_clock, INK, "ra")
        self._t(d, (W - 20, 54), f"of {_clock(self.duration)}  ·  frame {frame_idx}", self.f_small, MUTED, "ra")
        self._r(d, (20, 80, W - 20, 84), 2, fill=FAINT)
        prog = 20 + (W - 40) * min(1.0, t / max(self.duration, 1e-6))
        self._r(d, (20, 80, max(24, prog), 84), 2, fill=ORANGE)
        # KPI tiles
        on_screen = sum(1 for p in self.pids if frame_idx in self.tl[p]["frames"])
        kpis = [(str(on_screen), "players on screen"), (str(n_balls), "balls in view"),
                (str(n_unknown), "unsure boxes (?)")]
        kw = (W - 40 - 2 * 10) / 3
        for k, (v, lab) in enumerate(kpis):
            x0 = 20 + k * (kw + 10)
            self._r(d, (x0, 98, x0 + kw, 146), 10, fill=CARD)
            self._t(d, (x0 + 14, 104), v, self.f_kpi, ORANGE if k == 1 else INK)
            self._t(d, (x0 + 52, 116), lab, self.f_small, MUTED)
        # table header
        cols = {"player": 26, "status": 96, "on": 268, "ball": 352, "touch": 392, "dist": 522, "speed": 620}
        hy = 162
        self._t(d, (cols["player"], hy), "PLAYER", self.f_hdr, MUTED)
        self._t(d, (cols["status"], hy), "STATUS", self.f_hdr, MUTED)
        self._t(d, (cols["on"], hy), "ON SCREEN", self.f_hdr, MUTED, "ra")
        self._t(d, (cols["ball"], hy), "WITH BALL", self.f_hdr, MUTED, "ra")
        self._t(d, (cols["touch"], hy), "TOUCH", self.f_hdr, MUTED, "ma")
        self._t(d, (cols["dist"], hy), "DISTANCE", self.f_hdr, MUTED, "ra")
        self._t(d, (cols["speed"], hy), "SPEED", self.f_hdr, MUTED, "ra")
        y = 182
        rh = 40
        for p in self.pids:
            tl = self.tl[p]
            f = frame_idx
            on = f in tl["frames"]
            with_ball = f in tl["ball_frames"]
            touches = sum(1 for s in tl["touch_starts"] if s <= f)
            sp = tl["speed"][f] if f < len(tl["speed"]) else np.nan
            col = pid_color(p)
            self._r(d, (16, y, W - 16, y + rh - 4), 8, fill=CARD_ON if on else CARD)
            if on:
                self._r(d, (16, y, 21, y + rh - 4), 2, fill=col)
            txt = INK if on else MUTED
            cy = y + (rh - 4) / 2
            d.ellipse([(cols["player"] + 2) * SS, (cy - 7) * SS, (cols["player"] + 16) * SS, (cy + 7) * SS], fill=col)
            self._t(d, (cols["player"] + 24, cy), p.replace("PLAYER_", "P"), self.f_rowb, txt, "lm")
            # status pill
            if with_ball:
                self._r(d, (cols["status"], cy - 11, cols["status"] + 92, cy + 11), 11, fill=ORANGE)
                self._t(d, (cols["status"] + 46, cy), "WITH BALL", self.f_pill, BG, "mm")
            elif on:
                self._r(d, (cols["status"], cy - 11, cols["status"] + 92, cy + 11), 11, outline=GREEN, width=1)
                self._t(d, (cols["status"] + 46, cy), "ON COURT", self.f_pill, GREEN, "mm")
            else:
                self._t(d, (cols["status"] + 46, cy), "OFF SCREEN", self.f_pill, FAINT, "mm")
            # numbers
            self._t(d, (cols["on"], cy), f"{tl['vis'][f]:.1f}s", self.f_num, txt, "rm")
            bv = tl["ball"][f]
            self._t(d, (cols["ball"], cy - 3), f"{bv:.1f}s", self.f_num, ORANGE if bv > 0 else txt, "rm")
            bw = 64 * bv / self.max_ball
            self._r(d, (cols["ball"] - 64, cy + 10, cols["ball"], cy + 13), 1, fill=FAINT)
            if bw > 0.5:
                self._r(d, (cols["ball"] - 64, cy + 10, cols["ball"] - 64 + bw, cy + 13), 1, fill=ORANGE)
            self._t(d, (cols["touch"], cy), str(touches), self.f_num, txt, "mm")
            dv, rv = tl["dist"][f], tl["run"][f]
            self._t(d, (cols["dist"], cy - 3), f"{dv:.0f} m", self.f_num, txt, "rm")
            dw = 96 * dv / self.max_dist
            self._r(d, (cols["dist"] - 96, cy + 10, cols["dist"], cy + 13), 1, fill=FAINT)
            if dw > 0.5:
                self._r(d, (cols["dist"] - 96, cy + 10, cols["dist"] - 96 + dw, cy + 13), 1, fill=(120, 170, 230))
                rw = 96 * rv / self.max_dist
                if rw > 0.5:
                    self._r(d, (cols["dist"] - 96, cy + 10, cols["dist"] - 96 + rw, cy + 13), 1, fill=ORANGE)
            spd = f"{sp:.1f}" if on and not np.isnan(sp) else "—"
            self._t(d, (cols["speed"] - 26, cy), spd, self.f_num, txt, "rm")
            self._t(d, (cols["speed"], cy + 1), "m/s", self.f_small, MUTED, "rm")
            y += rh
        # legend for bars
        ly = y + 2
        self._r(d, (20, ly + 4, 34, ly + 8), 1, fill=(120, 170, 230)); self._t(d, (40, ly), "walk", self.f_small, MUTED)
        self._r(d, (82, ly + 4, 96, ly + 8), 1, fill=ORANGE); self._t(d, (102, ly), "run / ball time", self.f_small, MUTED)
        self._t(d, (W - 20, ly), "bars scaled to the clip maximum", self.f_small, FAINT, "ra")
        # mini-map
        my0, my1, mx0, mx1 = ly + 26, H - 82, 16, W - 16
        self._r(d, (mx0, my0, mx1, my1), 12, fill=CARD)
        for k in range(1, 4):
            gx = mx0 + (mx1 - mx0) * k / 4
            d.line([gx * SS, (my0 + 30) * SS, gx * SS, (my1 - 10) * SS], fill=(36, 42, 53), width=SS)
            gy = my0 + 30 + (my1 - my0 - 40) * k / 4
            d.line([(mx0 + 10) * SS, gy * SS, (mx1 - 10) * SS, gy * SS], fill=(36, 42, 53), width=SS)
        self._t(d, (mx0 + 14, my0 + 10), "TOP-DOWN MAP  ·  ESTIMATED", self.f_hdr, MUTED)
        self._t(d, (mx1 - 14, my0 + 10), "near camera at the bottom", self.f_small, FAINT, "ra")
        px0, px1, py0, py1 = mx0 + 20, mx1 - 20, my0 + 36, my1 - 16

        def to_map(x, z):
            return (px0 + (x - self.xlo) / (self.xhi - self.xlo) * (px1 - px0),
                    py1 - (z - self.zlo) / (self.zhi - self.zlo) * (py1 - py0))
        for p in self.pids:
            pos = self.tl[p]["pos"]
            trail = [to_map(*pos[k]) for k in range(max(0, frame_idx - 60), frame_idx + 1) if k in pos]
            if len(trail) >= 2:
                d.line([(a * SS, b * SS) for a, b in trail], fill=pid_color(p), width=2 * SS, joint="curve")
            if frame_idx in pos:
                cx, cy = to_map(*pos[frame_idx])
                c = pid_color(p)
                halo = tuple(int(v * 0.35 + BG[i] * 0.65) for i, v in enumerate(c))
                d.ellipse([(cx - 11) * SS, (cy - 11) * SS, (cx + 11) * SS, (cy + 11) * SS], fill=halo)
                d.ellipse([(cx - 6) * SS, (cy - 6) * SS, (cx + 6) * SS, (cy + 6) * SS], fill=c)
                self._t(d, (cx + 14, cy), p.replace("PLAYER_", "P"), self.f_rowb, INK, "lm")
        # footnotes
        a = self.meta["assumptions"]
        notes = ["IDs P01–P09 come from registration (no jersey numbers in this video).",
                 "Counted only while the ID is confirmed; UNKNOWN time is excluded.",
                 f"Distance & speed estimated without court calibration (±30%): height {a['player_height_m']} m, FOV {a['fov_long_side_deg']:.0f}°.",
                 "With ball = in hands, dribbling or overhead (warm-up: many have their own ball)."]
        for k, t_ in enumerate(notes):
            self._t(d, (20, H - 72 + 16 * k), t_, self.f_note, FAINT)
        img = img.resize((W, H), Image.LANCZOS)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _label(img, x1, y1, text, col_bgr, dark=(18, 20, 26)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 8, y_top + th + 8), col_bgr, -1)
    cv2.putText(img, text, (x1 + 4, y_top + th + 3), cv2.FONT_HERSHEY_DUPLEX, 0.45, dark, 1, cv2.LINE_AA)


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
    allpos = np.array([xy for p in pids for xy in timeline[p]["pos"].values()]) if pids else np.zeros((1, 2))
    xlo, xhi = np.percentile(allpos[:, 0], [1, 99]); zlo, zhi = np.percentile(allpos[:, 1], [1, 99])
    bounds = (xlo - 1, xhi + 1, max(0, zlo - 1), zhi + 1)
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     video], capture_output=True, text=True).stdout.strip() or 0)
    panel = None
    tmp = out_path + ".tmp.mp4"
    vw = None
    for fr in iter_frames(video):
        img = fr.image.copy()
        H, W = img.shape[:2]
        if panel is None:
            panel = Panel(H, pids, summary, timeline, meta, duration, bounds)
        n_unknown = 0
        for r in rows.get(fr.idx, []):
            x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
            if r["status"] == "CONFIRMED":
                col = pid_color(r["player_id"])[::-1]
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                _label(img, x1, y1, r["player_id"].replace("PLAYER_", "P"), col)
            elif r["status"] == "UNCERTAIN":
                n_unknown += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (150, 200, 230), 1, cv2.LINE_AA)
                _label(img, x1, y1, "?", (150, 200, 230))
        for b in balls.get(fr.idx, []):
            c = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
            cv2.circle(img, c, max(5, int((b[2] - b[0]) / 2) + 3), (36, 138, 240), 2, cv2.LINE_AA)
        side = panel.draw(fr.idx, fr.t, len(balls.get(fr.idx, [])), n_unknown)
        frame = np.hstack([img, side])
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", out_path], check=True)
    os.remove(tmp)
