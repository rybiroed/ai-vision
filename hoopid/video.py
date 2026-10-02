"""Streaming video reader with per-frame presentation timestamps.

Frames are decoded one at a time (never the whole video in RAM). OpenCV applies
the container's rotation matrix, so frames come out in display orientation.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    idx: int
    t: float  # presentation timestamp, seconds
    image: np.ndarray  # BGR, display orientation


def probe(path: str) -> dict:
    """Container/stream metadata via ffprobe (codec, rotation, fps, duration)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def frame_timestamps(path: str) -> list[float]:
    """Exact PTS of every video packet (the file is variable frame rate)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout
    ts = sorted(float(x) for x in out.split() if x.strip())
    return ts


def iter_frames(path: str, start: int = 0, end: int | None = None,
                timestamps: list[float] | None = None) -> Iterator[Frame]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while end is None or idx < end:
            ok, img = cap.read()
            if not ok:
                break
            if timestamps is not None and idx < len(timestamps):
                t = timestamps[idx]
            else:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield Frame(idx, t, img)
            idx += 1
    finally:
        cap.release()


def read_frame(path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise IndexError(idx)
    return img
