"""Review helpers: crop contact sheets per track (used for registration,
ground-truth labelling and error inspection)."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np

from .config import load_config
from .precompute import load
from .tracker import run_tracker
from .video import iter_frames


def collect_crops(video: str, pre: dict, wanted: list[int], pad: float = 0.08) -> dict[int, np.ndarray]:
    by_frame = defaultdict(list)
    for i in wanted:
        by_frame[int(pre["frame"][i])].append(i)
    out = {}
    last = max(by_frame) if by_frame else -1
    for fr in iter_frames(video, end=last + 1):
        for i in by_frame.get(fr.idx, []):
            x1, y1, x2, y2 = pre["box"][i]
            w, h = x2 - x1, y2 - y1
            H, W = fr.image.shape[:2]
            a, b = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
            c, d = int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
            out[i] = fr.image[b:d, a:c].copy()
    return out


def sheet(crops: list[tuple[str, np.ndarray]], cell_h: int = 128, per_row: int = 12) -> np.ndarray:
    cells = []
    for label, c in crops:
        s = cell_h / max(c.shape[0], 1)
        r = cv2.resize(c, (max(1, int(c.shape[1] * s)), cell_h), interpolation=cv2.INTER_NEAREST)
        cell = np.full((cell_h + 14, 72, 3), 255, np.uint8)
        w = min(72, r.shape[1])
        cell[14:, :w] = r[:, :w]
        cv2.putText(cell, label, (1, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        cells.append(cell)
    rows = []
    for k in range(0, len(cells), per_row):
        row = cells[k:k + per_row]
        row += [np.full_like(cells[0], 255)] * (per_row - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows) if rows else np.zeros((10, 10, 3), np.uint8)


def track_sheets(video: str, pre: dict, track_of: np.ndarray, out_dir: str, step: int = 10,
                 min_len: int = 5) -> list[int]:
    os.makedirs(out_dir, exist_ok=True)
    pick, tracks = [], {}
    for tid in np.unique(track_of[track_of >= 0]):
        ids = np.where(track_of == tid)[0]
        ids = ids[np.argsort(pre["frame"][ids])]
        if len(ids) < min_len:
            continue
        sel = list(ids[::step]) + ([ids[-1]] if (len(ids) - 1) % step else [])
        tracks[int(tid)] = sel
        pick += sel
    crops = collect_crops(video, pre, pick)
    for tid, sel in tracks.items():
        img = sheet([(f"{pre['frame'][i]}", crops[i]) for i in sel])
        cv2.putText(img, "", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(out_dir, f"track_{tid:04d}.jpg"), img)
    return sorted(tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--baseline-tracker", action="store_true")
    ap.add_argument("--step", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a.config)
    pre = load(a.pre)
    tcfg = dict(cfg["tracker"])
    if a.baseline_tracker:
        tcfg["appearance_gate"] = None
    tr = run_tracker(pre, tcfg)
    ids = track_sheets(a.video, pre, tr["track_of"], a.out, a.step)
    print(f"{len(ids)} track sheets in {a.out}")


if __name__ == "__main__":
    main()
