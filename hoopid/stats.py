"""Coach statistics per registered player.

Only frames where the player's identity is CONFIRMED are counted, so every
number is a lower bound for that player; UNKNOWN time is reported separately.

Metrics
  visible_s        time the player is on screen with a confirmed ID
  appearances      number of separate confirmed visible runs
  ball_s           time with a ball in hands / dribbling (ball detector + proximity)
  ball_touches     number of separate ball-control episodes
  dist_m           estimated distance covered (walk + run), m
  walk_m / run_m   distance at speed < run_speed / >= run_speed
  max_speed_ms     robust max speed (90th percentile of 1 s speeds)
  moving_pct       share of tracked time with speed > move_speed
  sprints          episodes >= sprint_min_s at speed >= sprint_speed

Distance and speed are ESTIMATES. With no court calibration the ground
position comes from a pinhole model: depth Z = f*H/h and lateral X = (u-cx)*H/h,
where h is the box height in pixels, H the player's height (registry field
`height_m`, else the configured default) and f the focal length from an
assumed field of view. Error sources: H, f, crouching/jumping, partial boxes.
Ball time on a warm-up / open-gym clip means "has a ball", not game possession.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .precompute import load

DEFAULTS = {
    "player_height_m": 1.75,
    "fov_long_side_deg": 65.0,      # phone main camera, long side of the frame (assumption)
    "pos_min_height_px": 45,
    "pos_max_occlusion": 0.25,
    "smooth_frames": 9,
    "step_frames": 5,
    "max_plausible_speed": 9.0,
    "move_speed": 0.7,
    "run_speed": 2.0,
    "sprint_speed": 4.5,
    "sprint_min_s": 0.7,
    "ball_min_score": 0.25,
    "ball_fill_gap": 6,
    "ball_merge_gap": 15,
    "ball_min_run": 3,
}


def _runs(frames: np.ndarray, max_gap: int) -> list[np.ndarray]:
    if len(frames) == 0:
        return []
    cut = np.where(np.diff(frames) > max_gap)[0] + 1
    return np.split(np.arange(len(frames)), cut)


def _median(x, k):
    if len(x) < 3:
        return x.copy()
    k = min(k, len(x) - (1 - len(x) % 2))
    r = k // 2
    xp = np.pad(x, r, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def ball_owner(pre_frame_rows: list[dict], balls: np.ndarray) -> dict[int, int]:
    """Map det_id -> 1 if that person controls a ball in this frame."""
    out = {}
    for bx in balls:
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        bd = max(bx[2] - bx[0], bx[3] - bx[1])
        cands = []
        for r in pre_frame_rows:
            x1, y1, x2, y2 = r["box"]
            w, h = x2 - x1, y2 - y1
            if bd > 0.6 * h:  # ball bigger than half a person: not this person's ball scale
                continue
            if x1 - 0.3 * w <= cx <= x2 + 0.3 * w and y1 - 0.25 * h <= cy <= y2 + 0.05 * h:  # incl. ball held overhead
                d = abs(cx - (x1 + x2) / 2) / w + 0.3 * abs(cy - (y1 + 0.55 * h)) / h
                cands.append((d, r["det_id"]))
        if not cands:
            continue
        cands.sort()
        if len(cands) > 1 and cands[1][0] - cands[0][0] < 0.25:
            continue  # two people equally close: do not attribute
        out[cands[0][1]] = 1
    return out


def compute(tracks_csv: str, pre_dir: str, ball_npz: str | None, registry: str | None, cfg: dict | None = None):
    c = dict(DEFAULTS); c.update(cfg or {})
    pre = load(pre_dir)
    pos = {int(d): k for k, d in enumerate(pre["det_id"])}
    t_of = dict(zip(pre["cam_frame"].tolist(), pre["cam_t"].tolist()))
    H_img, W_img = 848, 480
    long_side = max(H_img, W_img)
    f_px = (long_side / 2) / np.tan(np.radians(c["fov_long_side_deg"]) / 2)
    # camera drift compensation (cumulative translation, px)
    A = pre["cam_A"]
    cum_tx = np.cumsum(A[:, 0, 2]); cum_ty = np.cumsum(A[:, 1, 2])
    fidx = {int(f): k for k, f in enumerate(pre["cam_frame"])}
    heights = {}
    if registry and os.path.exists(registry):
        for pid, p in json.load(open(registry))["players"].items():
            if p.get("height_m"):
                heights[pid] = float(p["height_m"])
    rows = [r for r in csv.DictReader(open(tracks_csv)) if r["observed"] == "1" and r["det_id"]]
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append({"det_id": int(r["det_id"]),
                                          "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")]})
    has_ball = {}
    if ball_npz and os.path.exists(ball_npz):
        z = np.load(ball_npz)
        keep = z["score"] >= c["ball_min_score"]
        bf, bb = z["frame"][keep], z["box"][keep]
        for f in np.unique(bf):
            has_ball.update(ball_owner(by_frame.get(int(f), []), bb[bf == f]))
    per = defaultdict(list)
    unknown_frames = 0
    for r in rows:
        if r["status"] == "CONFIRMED":
            per[r["player_id"]].append(r)
        elif r["status"] in ("UNCERTAIN",):
            unknown_frames += 1
    n_frames = len(pre["cam_frame"])
    timeline = {}   # pid -> per-frame cumulative arrays for the dashboard
    summary = {}
    for pid, rr in sorted(per.items()):
        rr.sort(key=lambda r: int(r["frame"]))
        fr = np.array([int(r["frame"]) for r in rr])
        tt = np.array([t_of[f] for f in fr])
        Hm = heights.get(pid, c["player_height_m"])
        # visible time: sum of frame durations inside confirmed runs
        dt = np.diff(np.array(sorted(t_of.values())), append=sorted(t_of.values())[-1] + 1 / 30)
        dt_of = {f: dt[k] for k, f in enumerate(sorted(t_of))}
        vis_cum = np.zeros(n_frames)
        for f in fr:
            vis_cum[f] += dt_of[f]
        runs = _runs(fr, 3)
        # ball
        ball_flag = np.array([has_ball.get(int(r["det_id"]), 0) for r in rr], bool)
        ball_frames = set()
        for run in runs:
            fl = ball_flag[run]; ff = fr[run]
            on = np.where(fl)[0]
            if len(on) == 0:
                continue
            segs = _runs(ff[on], c["ball_fill_gap"])
            for s in segs:
                a, b = ff[on][s[0]], ff[on][s[-1]]
                if b - a + 1 >= c["ball_min_run"]:
                    ball_frames.update(range(a, b + 1))
        bfr = np.array(sorted(ball_frames))
        touches = len(_runs(bfr, c["ball_merge_gap"])) if len(bfr) else 0
        ball_cum = np.zeros(n_frames)
        for f in bfr:
            ball_cum[f] += dt_of.get(int(f), 1 / 30)
        # positions
        dist_cum = np.zeros(n_frames); run_cum = np.zeros(n_frames)
        speed_at = np.full(n_frames, np.nan)
        pos_xy = {}
        speeds_1s, moving_t, tracked_t, sprints = [], 0.0, 0.0, 0
        for run in runs:
            sel = []
            for k in run:
                i = pos[int(rr[k]["det_id"])]
                if pre["q_height"][i] >= c["pos_min_height_px"] and pre["q_occ"][i] <= c["pos_max_occlusion"] \
                        and not pre["q_trunc"][i]:
                    sel.append(k)
            if len(sel) < c["smooth_frames"]:
                continue
            sel = np.array(sel)
            f_s = fr[sel]
            box = np.array([[float(rr[k][x]) for x in ("x1", "y1", "x2", "y2")] for k in sel])
            u = (box[:, 0] + box[:, 2]) / 2 - np.array([cum_tx[fidx[f]] for f in f_s])
            h = _median(box[:, 3] - box[:, 1], c["smooth_frames"] * 2 + 1)
            u = _median(u, c["smooth_frames"])
            Z = f_px * Hm / h
            X = (u - W_img / 2) * Hm / h
            k = c["smooth_frames"]
            ker = np.ones(k) / k
            if len(X) > k:
                X = np.convolve(np.pad(X, k // 2, mode="edge"), ker, "valid")[:len(f_s)]
                Z = np.convolve(np.pad(Z, k // 2, mode="edge"), ker, "valid")[:len(f_s)]
            for j, f in enumerate(f_s):
                pos_xy[int(f)] = (float(X[j]), float(Z[j]))
            step = c["step_frames"]
            idx = np.arange(0, len(f_s), step)
            if idx[-1] != len(f_s) - 1:
                idx = np.append(idx, len(f_s) - 1)
            for a, b in zip(idx, idx[1:]):
                d = float(np.hypot(X[b] - X[a], Z[b] - Z[a]))
                dts = t_of[int(f_s[b])] - t_of[int(f_s[a])]
                if dts <= 0:
                    continue
                v = d / dts
                if v > c["max_plausible_speed"]:
                    continue  # box jump, not motion
                dist_cum[f_s[b]] += d
                if v >= c["run_speed"]:
                    run_cum[f_s[b]] += d
                speed_at[f_s[a]:f_s[b] + 1] = v
                tracked_t += dts
                if v > c["move_speed"]:
                    moving_t += dts
                speeds_1s.append(v)
            # sprints: consecutive steps over sprint speed lasting >= sprint_min_s
            sp = [(t_of[int(f_s[a])], t_of[int(f_s[b])], np.hypot(X[b] - X[a], Z[b] - Z[a]) /
                   max(t_of[int(f_s[b])] - t_of[int(f_s[a])], 1e-6)) for a, b in zip(idx, idx[1:])]
            start = None
            for t0, t1, v in sp + [(None, None, 0)]:
                if v >= c["sprint_speed"] and v <= c["max_plausible_speed"]:
                    start = t0 if start is None else start; end = t1
                else:
                    if start is not None and end - start >= c["sprint_min_s"]:
                        sprints += 1
                    start = None
        tot = float(dist_cum.sum()); run_m = float(run_cum.sum())
        summary[pid] = {
            "visible_s": round(float(vis_cum.sum()), 1), "appearances": len(runs),
            "ball_s": round(float(ball_cum.sum()), 1), "ball_touches": touches,
            "dist_m": round(tot, 1), "walk_m": round(tot - run_m, 1), "run_m": round(run_m, 1),
            "max_speed_ms": round(float(np.percentile(speeds_1s, 90)), 2) if speeds_1s else None,
            "moving_pct": round(100 * moving_t / tracked_t, 0) if tracked_t else None,
            "sprints": sprints, "height_m_used": Hm,
        }
        timeline[pid] = {"vis": np.cumsum(vis_cum), "ball": np.cumsum(ball_cum), "dist": np.cumsum(dist_cum),
                         "run": np.cumsum(run_cum), "speed": speed_at, "pos": pos_xy,
                         "frames": set(fr.tolist()), "ball_frames": ball_frames,
                         "touch_starts": [int(r[0]) for r in [bfr[x] for x in _runs(bfr, c["ball_merge_gap"])]] if len(bfr) else []}
    meta = {"f_px": round(float(f_px), 1), "assumptions": {k: c[k] for k in ("player_height_m", "fov_long_side_deg")},
            "unknown_detections": unknown_frames, "balls_attributed": len(has_ball), "config": c}
    return summary, timeline, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pre", default="runs/full/precompute")
    ap.add_argument("--ball", default="runs/full/ball.npz")
    ap.add_argument("--registry", default="match/players.json")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--video", default=None, help="render dashboard video")
    a = ap.parse_args()
    s, tl, meta = compute(a.tracks, a.pre, a.ball, a.registry)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"players": s, "meta": meta}, open(os.path.join(a.out, "coach_stats.json"), "w"), indent=2,
              ensure_ascii=False, default=float)
    with open(os.path.join(a.out, "coach_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(next(iter(s.values())).keys())
        w.writerow(["player_id"] + keys)
        for pid, v in s.items():
            w.writerow([pid] + [v[k] for k in keys])
    print(json.dumps(s, indent=1))
    if a.video:
        from .dashboard import render
        render(a.video, a.tracks, tl, s, meta, os.path.join(a.out, "coach_dashboard.mp4"), a.ball)


if __name__ == "__main__":
    main()
