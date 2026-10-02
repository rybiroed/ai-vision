"""Short-term multi-object tracker (ByteTrack-style, own implementation).

* Two-stage association: confident detections first, then low-score ones.
* Constant-velocity Kalman filter on box centre and size.
* Camera motion compensation: predicted boxes are warped by the per-frame
  similarity transform from `camera.MotionEstimator`. If the motion estimate is
  unreliable the prediction is not trusted (gating is tightened to pure IoU of
  the last observed box and the velocity is zeroed).
* A detected hard cut terminates every track: nothing is carried across a cut
  by position.
* Optional appearance gate (Re-ID cosine) refuses IoU matches whose appearance
  disagrees with the track; this reduces swaps inside a tracklet when two
  players cross. The plain baseline runs with the gate disabled.

Output: one track_id per detection (-1 = unassigned) and, for every frame a
track is alive but unobserved, a LOST record (predicted box flagged as a
prediction, never as an observation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera import warp_points


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


class Kalman:
    """State: cx, cy, w, h, vcx, vcy, vw, vh."""

    def __init__(self, box: np.ndarray):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float64)
        s = max(h, 10.0)
        self.P = np.diag([(0.1 * s) ** 2] * 4 + [(0.3 * s) ** 2] * 4)
        self.F = np.eye(8); self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self):
        s = max(self.x[3], 10.0)
        Q = np.diag([(0.05 * s) ** 2] * 4 + [(0.0125 * s) ** 2] * 4)
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 2)
        self.P = self.F @ self.P @ self.F.T + Q

    def warp(self, A: np.ndarray):
        sc = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
        self.x[:2] = warp_points(A, self.x[None, :2])[0]
        self.x[4:6] = A[:, :2] @ self.x[4:6]
        self.x[2:4] *= sc; self.x[6:8] *= sc

    def freeze_motion(self):
        self.x[4:] = 0
        self.P[4:, 4:] *= 4

    def update(self, box: np.ndarray):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        s = max(z[3], 10.0)
        R = np.diag([(0.05 * s) ** 2] * 4)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    tid: int
    kf: Kalman
    start: int
    last_obs: int
    last_box: np.ndarray
    hits: int = 1
    emb: np.ndarray | None = None
    lost_frames: int = 0
    det_ids: list = field(default_factory=list)


class Tracker:
    def __init__(self, cfg: dict):
        self.high = cfg["high_thr"]
        self.low = cfg["low_thr"]
        self.new = cfg["new_track_thr"]
        self.iou1 = cfg["match_iou"]
        self.iou2 = cfg["match_iou_low"]
        self.max_age = cfg["max_lost_frames"]
        self.min_h = cfg.get("min_box_height", 0)
        self.app_gate = cfg.get("appearance_gate")  # None disables
        self.app_gate_min_h = cfg.get("appearance_gate_min_height", 60)
        self.ema = cfg.get("embedding_ema", 0.9)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.finished: list[Track] = []
        self.events: list[dict] = []

    def _terminate_all(self, frame: int, reason: str):
        for t in self.tracks:
            self.events.append({"frame": frame, "track_id": t.tid, "event": "terminated", "reason": reason})
        self.finished += self.tracks
        self.tracks = []

    def step(self, frame: int, boxes: np.ndarray, scores: np.ndarray, det_ids: np.ndarray,
             embs: np.ndarray, heights: np.ndarray, cam: dict) -> tuple[dict, list]:
        """Returns {det_id: track_id} and a list of (track_id, predicted_box) LOST rows."""
        if cam["cut"]:
            self._terminate_all(frame, "shot_cut")
        for t in self.tracks:
            t.kf.predict()
            if cam["reliable"]:
                t.kf.warp(cam["A"])
            else:
                t.kf.freeze_motion()
        keep = (scores >= self.low) & (heights >= self.min_h)
        boxes, scores, det_ids, embs, heights = boxes[keep], scores[keep], det_ids[keep], embs[keep], heights[keep]
        hi = scores >= self.high
        assign: dict[int, int] = {}

        def match(track_idx: list[int], det_idx: np.ndarray, thr: float) -> tuple[list, list]:
            if not track_idx or len(det_idx) == 0:
                return track_idx, list(det_idx)
            tb = np.stack([self.tracks[i].kf.box() if cam["reliable"] else
                           (self.tracks[i].kf.box() + self.tracks[i].last_box) / 2 for i in track_idx])
            iou = iou_matrix(tb, boxes[det_idx])
            cost = 1 - iou
            if self.app_gate is not None:
                for a, ti in enumerate(track_idx):
                    e = self.tracks[ti].emb
                    if e is None:
                        continue
                    sims = embs[det_idx] @ e
                    bad = (sims < self.app_gate) & (heights[det_idx] >= self.app_gate_min_h)
                    cost[a, bad] = 1.0
            r, c = linear_sum_assignment(cost)
            mt, md = set(), set()
            for a, b in zip(r, c):
                if 1 - cost[a, b] >= thr:
                    ti, di = track_idx[a], det_idx[b]
                    t = self.tracks[ti]
                    t.kf.update(boxes[di]); t.last_obs = frame; t.last_box = boxes[di].copy()
                    t.hits += 1; t.lost_frames = 0
                    e = embs[di].astype(np.float32)
                    if heights[di] >= self.app_gate_min_h:
                        t.emb = e if t.emb is None else self.ema * t.emb + (1 - self.ema) * e
                        t.emb /= np.linalg.norm(t.emb) + 1e-9
                    t.det_ids.append(int(det_ids[di]))
                    assign[int(det_ids[di])] = t.tid
                    mt.add(ti); md.add(di)
            return [i for i in track_idx if i not in mt], [d for d in det_idx if d not in md]

        all_t = list(range(len(self.tracks)))
        rem_t, rem_hi = match(all_t, np.where(hi)[0], self.iou1)
        rem_t, _ = match(rem_t, np.where(~hi)[0], self.iou2)
        for di in rem_hi:
            if scores[di] < self.new:
                continue
            t = Track(self.next_id, Kalman(boxes[di]), frame, frame, boxes[di].copy())
            if heights[di] >= self.app_gate_min_h:
                t.emb = embs[di].astype(np.float32)
            t.det_ids.append(int(det_ids[di]))
            assign[int(det_ids[di])] = t.tid
            self.events.append({"frame": frame, "track_id": t.tid, "event": "started"})
            self.next_id += 1
            self.tracks.append(t)
        lost_rows, alive = [], []
        for i, t in enumerate(self.tracks):
            if t.last_obs == frame:
                alive.append(t); continue
            t.lost_frames += 1
            if t.lost_frames > self.max_age:
                self.events.append({"frame": frame, "track_id": t.tid, "event": "terminated",
                                    "reason": f"unobserved>{self.max_age}f"})
                self.finished.append(t)
            else:
                lost_rows.append((t.tid, t.kf.box()))
                alive.append(t)
        self.tracks = alive
        return assign, lost_rows


def run_tracker(pre: dict, cfg: dict) -> dict:
    """Run the tracker over precomputed detections. Returns arrays aligned to detections."""
    tr = Tracker(cfg)
    frames = pre["frame"]
    order = np.argsort(frames, kind="stable")
    starts = np.searchsorted(frames[order], pre["cam_frame"])
    ends = np.searchsorted(frames[order], pre["cam_frame"], side="right")
    track_of = np.full(len(frames), -1, np.int64)
    pos = {int(d): i for i, d in enumerate(pre["det_id"])}
    lost = []
    reid = pre["reid"].astype(np.float32)
    for k, f in enumerate(pre["cam_frame"]):
        idx = order[starts[k]:ends[k]]
        cam = {"A": pre["cam_A"][k], "reliable": bool(pre["cam_reliable"][k]), "cut": bool(pre["cam_cut"][k])}
        assign, lrows = tr.step(int(f), pre["box"][idx], pre["score"][idx], pre["det_id"][idx],
                                reid[idx], pre["q_height"][idx], cam)
        for d, t in assign.items():
            track_of[pos[d]] = t
        lost += [(int(f), tid, b) for tid, b in lrows]
    return {"track_of": track_of, "lost": lost, "events": tr.events}
