"""Manual corrections (corrections.json) and their application.

Operations (all logged with a timestamp and source; the UI also records how
long the operator spent, so correction time can be measured in real use):

  {"op": "assign",  "track_id": T, "frames": [a, b], "player_id": "PLAYER_03"}
  {"op": "unknown", "track_id": T, "frames": [a, b]}
  {"op": "split",   "track_id": T, "frame": F}
  {"op": "merge",   "track_ids": [T1, T2], "player_id": "PLAYER_03"}   # = assign on both whole tracks
  {"op": "set_attr","player_id": "PLAYER_03", "team": "white", "number": "7"}

An assignment only covers the frame range given. It is never propagated over
the rest of the track automatically, because a track may contain several
people. After editing, re-run `hoopid.pipeline` (seconds; detections are
cached): only identity decisions are recomputed.
"""
from __future__ import annotations

import datetime as dt
import json
import os


def load(path: str) -> list[dict]:
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return []


def save(path: str, ops: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ops, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def add(path: str, op: dict, source: str = "user", seconds_spent: float | None = None) -> list[dict]:
    ops = load(path)
    op = dict(op)
    op["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    op["source"] = source
    if seconds_spent is not None:
        op["seconds_spent"] = round(float(seconds_spent), 1)
    if op["op"] == "merge":
        # expand into explicit whole-track assigns so the identity manager sees plain locks
        ops.append(op)
    else:
        ops.append(op)
    save(path, ops)
    return ops


def expand(ops: list[dict], track_ranges: dict[int, tuple[int, int]]) -> list[dict]:
    """Turn merge ops into assign ops over the merged tracks' full ranges."""
    out = []
    for o in ops:
        if o["op"] == "merge":
            for t in o["track_ids"]:
                if int(t) in track_ranges:
                    a, b = track_ranges[int(t)]
                    out.append({"op": "assign", "track_id": int(t), "frames": [a, b],
                                "player_id": o["player_id"], "from": "merge"})
        elif o["op"] in ("assign", "unknown", "split"):
            out.append(o)
    return out
