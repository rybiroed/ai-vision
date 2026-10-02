"""Exports: tracks.csv, assignments.jsonl, annotated video."""
from __future__ import annotations

import csv
import json
import os
import subprocess

import cv2
import numpy as np

from .video import iter_frames

STATUS_COLOR = {"CONFIRMED": (60, 200, 60), "UNCERTAIN": (0, 200, 255),
                "UNREGISTERED": (160, 160, 160), "LOST": (0, 0, 255)}


def _pid_color(pid: str) -> tuple:
    if not pid.startswith("PLAYER_"):
        return (200, 200, 200)
    h = (int(pid.split("_")[1]) * 37) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


def write_tracks_csv(path: str, pre: dict, track_of: np.ndarray, final: dict, online: dict,
                     lost: list, seg_player_at: dict | None = None) -> None:
    """One row per observed detection (observed=1) plus LOST rows (observed=0).

    LOST rows carry the tracker's *predicted* box, flagged predicted=1; they are
    never counted as observations.
    """
    t_of = dict(zip(pre["cam_frame"].tolist(), pre["cam_t"].tolist()))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "det_id", "track_id", "segment", "observed", "predicted",
                    "x1", "y1", "x2", "y2", "det_score", "player_id", "status", "id_score", "id_margin",
                    "n_eff", "candidates", "player_id_online", "status_online", "decision_reason"])
        rows = []
        for i in range(len(pre["frame"])):
            f = int(pre["frame"][i])
            r = final.get(i)
            o = online.get(i, {})
            tid = int(track_of[i])
            if r is None:
                r = {"status": "UNTRACKED", "player_id": "", "score": "", "margin": "", "n_eff": "",
                     "candidates": [], "segment": -1}
            b = pre["box"][i]
            rows.append([f, round(t_of.get(f, 0.0), 4), int(pre["det_id"][i]), tid, r.get("segment", -1), 1, 0,
                         *[round(float(v), 1) for v in b], round(float(pre["score"][i]), 3), r["player_id"],
                         r["status"], r["score"], r["margin"], r["n_eff"], json.dumps(r["candidates"]),
                         o.get("player_id", ""), o.get("status", ""), r.get("reason", "")])
        for f, tid, b in lost:
            pid = (seg_player_at or {}).get((tid, f))
            if not pid:
                continue
            rows.append([f, round(t_of.get(f, 0.0), 4), "", tid, "", 0, 1, *[round(float(v), 1) for v in b],
                         "", pid, "LOST", "", "", "", "[]", "", "", "tracker_lost_prediction"])
        rows.sort(key=lambda r: (r[0], str(r[3])))
        w.writerows(rows)


def lost_player_map(pre: dict, track_of: np.ndarray, final: dict, lost: list) -> dict:
    """(track_id, frame) -> player_id for frames where a CONFIRMED track is unobserved."""
    last_pid: dict[int, list] = {}
    order = np.argsort(pre["frame"], kind="stable")
    for i in order:
        r = final.get(int(i))
        if r and track_of[i] >= 0:
            last_pid.setdefault(int(track_of[i]), []).append((int(pre["frame"][i]), r["player_id"], r["status"]))
    out = {}
    for f, tid, _ in lost:
        hist = last_pid.get(tid, [])
        prev = [h for h in hist if h[0] < f]
        if prev and prev[-1][2] == "CONFIRMED":
            out[(tid, f)] = prev[-1][1]
    return out


def write_assignments(path: str, log: list[dict], corrections: list[dict]) -> None:
    with open(path, "w") as fh:
        for c in corrections:
            fh.write(json.dumps({"type": "correction", **c}, ensure_ascii=False) + "\n")
        for e in log:
            e = dict(e)
            e["pass"] = e.pop("pass_", None)
            fh.write(json.dumps({"type": "decision", **e}, ensure_ascii=False, default=str) + "\n")


def render_video(video: str, out_path: str, pre: dict, track_of: np.ndarray, final: dict,
                 lost: list, lost_map: dict, title: str = "") -> None:
    by_frame: dict[int, list] = {}
    for i in range(len(pre["frame"])):
        by_frame.setdefault(int(pre["frame"][i]), []).append(i)
    lost_by_frame: dict[int, list] = {}
    for f, tid, b in lost:
        if (tid, f) in lost_map:
            lost_by_frame.setdefault(f, []).append((lost_map[(tid, f)], b))
    tmp = out_path + ".tmp.mp4"
    vw = None
    fps = 30.0
    for fr in iter_frames(video):
        img = fr.image.copy()
        if vw is None:
            h, w = img.shape[:2]
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in by_frame.get(fr.idx, []):
            r = final.get(i)
            x1, y1, x2, y2 = [int(v) for v in pre["box"][i]]
            if r is None:
                cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
                continue
            st, pid = r["status"], r["player_id"]
            col = _pid_color(pid) if st == "CONFIRMED" else STATUS_COLOR.get(st, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if st == "CONFIRMED" else 1)
            label = pid.replace("PLAYER_", "P") if st == "CONFIRMED" else ("?" if st == "UNCERTAIN" else "x")
            label += f" t{int(track_of[i])}"
            cv2.putText(img, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        for pid, b in lost_by_frame.get(fr.idx, []):
            x1, y1, x2, y2 = [int(v) for v in b]
            # dashed outline: a prediction, not an observation
            for x in range(x1, x2, 8):
                cv2.line(img, (x, y1), (min(x + 4, x2), y1), STATUS_COLOR["LOST"], 1)
                cv2.line(img, (x, y2), (min(x + 4, x2), y2), STATUS_COLOR["LOST"], 1)
            cv2.putText(img, pid.replace("PLAYER_", "P") + " LOST", (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, STATUS_COLOR["LOST"], 1)
        cv2.putText(img, f"{title} f{fr.idx} {fr.t:.2f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(img, f"{title} f{fr.idx} {fr.t:.2f}s", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        vw.write(img)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", out_path], check=True)
    os.remove(tmp)
