"""Ball detection pass: YOLOX COCO class 32 ("sports ball"), streamed, saved to npz.

A basketball is 10-25 px in this footage, so besides the full frame the
detector also runs on square tiles of the band where players are, at 2x
(tiles are only a detector input scale, nothing is "restored").

    python -m hoopid.ball --video V --out runs/full/ball.npz --tiles 320 --band-y 230
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from .detect import YoloxDetector, _nms
from .video import iter_frames

SPORTS_BALL = 32


def detect_balls(det: YoloxDetector, img: np.ndarray, tile: int, band_y: int, full: bool = True) -> np.ndarray:
    H, W = img.shape[:2]
    bs, ss = [], []
    if full:
        b, s = det._run(img); bs.append(b); ss.append(s)
    if tile:
        xs = np.linspace(0, W - tile, int(np.ceil(W / tile)) + (W % tile > 0)).astype(int) if W > tile else [0]
        for x0 in sorted(set(int(x) for x in xs)):
            b, s = det._run(img[band_y:band_y + tile, x0:x0 + tile])
            b[:, [0, 2]] += x0; b[:, [1, 3]] += band_y
            bs.append(b); ss.append(s)
    b = np.concatenate(bs); s = np.concatenate(ss)
    keep = _nms(b, s, 0.3)
    return np.concatenate([b[keep], s[keep, None]], 1) if keep else np.zeros((0, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/yolox_m.onnx")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--tiles", type=int, default=320, help="tile size (0 = full frame only)")
    ap.add_argument("--band-y", type=int, default=230, help="top of the tile band (players' zone)")
    ap.add_argument("--no-full", action="store_true", help="skip the full-frame pass")
    ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    det = YoloxDetector(a.model, score_thr=a.thr, nms_thr=0.4, cls=SPORTS_BALL)
    frames, boxes = [], []
    t0 = time.time()
    n = 0
    for fr in iter_frames(a.video, end=a.end):
        b = detect_balls(det, fr.image, a.tiles, a.band_y, not a.no_full)
        frames.append(np.full(len(b), fr.idx, np.int32)); boxes.append(b.astype(np.float32).reshape(-1, 5))
        n += 1
        if n % 300 == 0:
            print(f"[ball] {n} frames, {sum(len(x) for x in boxes)} balls, {n / (time.time() - t0):.2f} fps", flush=True)
    f = np.concatenate(frames); b = np.concatenate(boxes)
    np.savez_compressed(a.out, frame=f, box=b[:, :4], score=b[:, 4])
    print(json.dumps({"frames": n, "balls": int(len(f)), "wall_s": round(time.time() - t0, 1),
                      "tiles": a.tiles, "band_y": a.band_y, "full": not a.no_full}))


def merge(paths: list[str], out: str):
    fs, bs, ss = [], [], []
    for p in paths:
        z = np.load(p); fs.append(z["frame"]); bs.append(z["box"]); ss.append(z["score"])
    f = np.concatenate(fs); b = np.concatenate(bs); s = np.concatenate(ss)
    of, ob, os_ = [], [], []
    for fr in np.unique(f):
        m = f == fr
        keep = _nms(b[m], s[m], 0.3)
        of.append(np.full(len(keep), fr)); ob.append(b[m][keep]); os_.append(s[m][keep])
    np.savez_compressed(out, frame=np.concatenate(of), box=np.concatenate(ob), score=np.concatenate(os_))


if __name__ == "__main__":
    main()
