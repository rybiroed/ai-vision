"""Camera motion compensation (CMC) and shot-cut detection.

For each frame we estimate a 2x3 similarity transform mapping pixel
coordinates of the previous frame to the current one, using ORB features on
the static background (detected people are masked out). If too few inliers are
found, the motion estimate is flagged unreliable and the tracker must not
trust its motion prediction across that frame. A colour-histogram jump flags a
hard cut; after a cut, no track is carried over by position.
"""
from __future__ import annotations

import cv2
import numpy as np

SCALE = 0.5


class MotionEstimator:
    def __init__(self, min_inliers: int = 25, cut_corr: float = 0.55):
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev = None
        self.prev_hist = None
        self.min_inliers = min_inliers
        self.cut_corr = cut_corr

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> dict:
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        mask = np.full(gray.shape, 255, np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (b[:4] * SCALE).astype(int)
            mask[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
        kp, des = self.orb.detectAndCompute(gray, mask)
        res = {"A": np.array([[1, 0, 0], [0, 1, 0]], np.float32), "inliers": 0,
               "reliable": False, "cut": False, "hist_corr": 1.0}
        if self.prev is not None:
            corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
            res["hist_corr"] = corr
            res["cut"] = corr < self.cut_corr
            pkp, pdes = self.prev
            if des is not None and pdes is not None and len(kp) > 10 and len(pkp) > 10:
                m = self.bf.match(pdes, des)
                if len(m) >= 10:
                    src = np.float32([pkp[x.queryIdx].pt for x in m]) / SCALE
                    dst = np.float32([kp[x.trainIdx].pt for x in m]) / SCALE
                    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                         ransacReprojThreshold=3.0)
                    if A is not None:
                        ni = int(inl.sum())
                        res["inliers"] = ni
                        if ni >= self.min_inliers:
                            res["A"] = A.astype(np.float32)
                            res["reliable"] = True
        self.prev = (kp, des)
        self.prev_hist = hist
        return res


def warp_points(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ A[:, :2].T + A[:, 2]
