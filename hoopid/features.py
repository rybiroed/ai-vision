"""Per-detection appearance features and quality measures.

All crops are taken from the original-resolution frame. Nothing here is
upscaled to "recover" detail: if a region is too small, the feature is marked
unavailable (NaN / zero weight) instead of being guessed.

Features
  reid    : 768-d YoutuReID embedding (OpenCV Zoo, Apache-2.0), L2-normalised
  torso   : HSV histogram of the shirt region
  legs    : HSV histogram of the shorts/upper-leg region
  shoes   : HSV histogram of the shoe strip with floor-coloured pixels removed
  hair    : HSV histogram of the top of the head, only for large boxes
Quality
  height, truncated (touches frame edge), occlusion (fraction of the box covered
  by other boxes), sharpness (variance of Laplacian), shoe_valid/hair_valid.
Visible skin tone is deliberately NOT extracted: on this footage it is
dominated by lighting, it carries a risk of being used as a demographic proxy,
and the prompt allows it only as a weak auxiliary cue.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

H_BINS, S_BINS, V_BINS = 12, 4, 3
HIST_DIM = H_BINS * S_BINS * V_BINS
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# region bands as fractions of box height
REGIONS = {
    "hair": (0.00, 0.07),
    "torso": (0.17, 0.45),
    "legs": (0.47, 0.68),
    "shoes": (0.90, 1.00),
}
MIN_H_HAIR = 140  # px box height for the hair band to be ~10px tall
MIN_H_SHOES = 80


class ReidModel:
    def __init__(self, path: str, threads: int = 0, providers=None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=providers or ort.get_available_providers())
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), np.float32)
        batch = []
        for c in crops:
            c = cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR)
            c = (c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
            batch.append(c.transpose(2, 0, 1))
        out = self.sess.run(None, {self.inp: np.stack(batch)})[0].reshape(len(crops), -1)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def _hist(hsv: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if hsv.size == 0 or (mask is not None and mask.sum() < 12):
        return np.full(HIST_DIM, np.nan, np.float32)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    h = h.flatten().astype(np.float32)
    return h / (h.sum() + 1e-9)


def _floor_mask(hsv_band: np.ndarray, floor_ref: np.ndarray) -> np.ndarray:
    """1 where pixel is NOT floor-like. floor_ref: Kx3 HSV samples beside the feet."""
    if len(floor_ref) < 8:
        return np.full(hsv_band.shape[:2], 255, np.uint8)
    lab_like = hsv_band.reshape(-1, 3).astype(np.float32)
    ref = floor_ref.astype(np.float32)
    mu = np.median(ref, 0)
    dev = np.maximum(np.median(np.abs(ref - mu), 0) * 3.0, [8, 25, 30])
    dh = np.abs(lab_like[:, 0] - mu[0]); dh = np.minimum(dh, 180 - dh)
    floorish = (dh < dev[0]) & (np.abs(lab_like[:, 1] - mu[1]) < dev[1]) & (np.abs(lab_like[:, 2] - mu[2]) < dev[2])
    return np.where(floorish, 0, 255).astype(np.uint8).reshape(hsv_band.shape[:2])


def occlusion(boxes: np.ndarray) -> np.ndarray:
    """Fraction of each box's area covered by the union of boxes in front of it.

    A box whose bottom edge is lower in the image is assumed to be closer to the
    camera (in front). Coverage is approximated by the max single overlap.
    """
    n = len(boxes)
    occ = np.zeros(n, np.float32)
    for i in range(n):
        ax1, ay1, ax2, ay2 = boxes[i, :4]
        area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
        for j in range(n):
            if i == j or boxes[j, 3] <= boxes[i, 3]:
                continue
            ix = max(0.0, min(ax2, boxes[j, 2]) - max(ax1, boxes[j, 0]))
            iy = max(0.0, min(ay2, boxes[j, 3]) - max(ay1, boxes[j, 1]))
            occ[i] = max(occ[i], ix * iy / area)
    return occ


def extract(frame: np.ndarray, boxes: np.ndarray, reid: ReidModel | None) -> dict[str, np.ndarray]:
    H, W = frame.shape[:2]
    n = len(boxes)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = {k: np.full((n, HIST_DIM), np.nan, np.float32) for k in REGIONS}
    q_height = np.zeros(n, np.float32)
    q_trunc = np.zeros(n, np.uint8)
    q_sharp = np.zeros(n, np.float32)
    crops = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, max(x2, x1 + 2)), min(H, max(y2, y1 + 2))
        h = y2 - y1
        q_height[i] = h
        q_trunc[i] = int(x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1)
        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
        q_sharp[i] = cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_32F).var()
        # restrict colour sampling to the central 60% of the box width (less background)
        cx1 = x1 + int(0.2 * (x2 - x1)); cx2 = x2 - int(0.2 * (x2 - x1))
        for name, (f0, f1) in REGIONS.items():
            if name == "hair" and h < MIN_H_HAIR:
                continue
            if name == "shoes" and (h < MIN_H_SHOES or q_trunc[i]):
                continue
            ry1, ry2 = y1 + int(f0 * h), y1 + max(int(f1 * h), int(f0 * h) + 2)
            if name == "shoes":
                band = hsv[ry1:ry2, x1:x2]
                # floor reference: strips immediately left/right of the box at foot level
                lw = max(4, (x2 - x1) // 3)
                ref = np.concatenate([
                    hsv[ry1:ry2, max(0, x1 - lw):x1].reshape(-1, 3),
                    hsv[ry1:ry2, x2:min(W, x2 + lw)].reshape(-1, 3)])
                mask = _floor_mask(band, ref)
                out[name][i] = _hist(band, mask)
            else:
                band = hsv[ry1:ry2, cx1:cx2]
                out[name][i] = _hist(band, None)
    feats = dict(out)
    feats["reid"] = (reid(crops).astype(np.float16) if reid is not None and n
                     else np.zeros((n, 768), np.float16))
    feats["q_height"] = q_height
    feats["q_trunc"] = q_trunc
    feats["q_sharp"] = q_sharp
    feats["q_occ"] = occlusion(boxes) if n else np.zeros(0, np.float32)
    return feats


def hist_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya similarity in [0,1]; NaN if either side is unavailable."""
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
