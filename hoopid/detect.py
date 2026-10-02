"""Person detector: YOLOX (Megvii, Apache-2.0) COCO weights, ONNX Runtime.

Only the COCO "person" class is kept. Optional tiling runs the detector on
overlapping square tiles so that distant players are seen at ~1x scale instead
of being shrunk by the 640x640 letterbox.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import cv2

PERSON = 0


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    idx = cv2.dnn.NMSBoxes(
        [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
        scores.astype(float).tolist(), 0.0, thr)
    return [int(i) for i in np.array(idx).reshape(-1)]


class YoloxDetector:
    def __init__(self, model_path: str, input_size: int = 640, score_thr: float = 0.3,
                 nms_thr: float = 0.5, tiles: str = "none", threads: int = 0,
                 providers: list[str] | None = None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(
            model_path, so, providers=providers or ort.get_available_providers())
        self.inp = self.sess.get_inputs()[0].name
        self.size = input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.tiles = tiles
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def _run(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[: int(h * r), : int(w * r)] = cv2.resize(
            img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        score = out[:, 4] * out[:, 5 + PERSON]
        keep = score >= self.score_thr
        xy, wh, score = xy[keep], wh[keep], score[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        return boxes, score

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Returns Nx5 array [x1, y1, x2, y2, score] in image pixels."""
        h, w = img.shape[:2]
        all_b, all_s = [], []
        b, s = self._run(img)
        all_b.append(b); all_s.append(s)
        if self.tiles == "vertical" and h > w:
            # two/three square tiles stacked vertically with overlap
            n = int(np.ceil((h - w) / (w * 0.75))) + 1
            ys = np.linspace(0, h - w, n).astype(int)
            for y0 in ys:
                tb, ts = self._run(img[y0:y0 + w])
                tb[:, [1, 3]] += y0
                # drop boxes cut by an internal tile edge; the full-frame pass covers them
                cut = ((tb[:, 1] < y0 + 2) & (y0 > 0)) | ((tb[:, 3] > y0 + w - 2) & (y0 + w < h))
                all_b.append(tb[~cut]); all_s.append(ts[~cut])
        boxes = np.concatenate(all_b)
        scores = np.concatenate(all_s)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        keep = _nms(boxes, scores, self.nms_thr)
        return np.concatenate([boxes[keep], scores[keep, None]], 1) if keep else np.zeros((0, 5))
