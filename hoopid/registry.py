"""Player registry (players.json) for one match.

Each player has a team/number (optional), a list of user-confirmed samples and
a history of edits. A sample points at a detection of the precomputed run
(det_id) so its features are exactly those the matcher sees. Samples carry
quality metadata and the view (front/back/side) chosen by the user.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):
            with open(path) as fh:
                self.data = json.load(fh)
        else:
            self.data = {"match": None, "players": {}}

    @property
    def players(self) -> dict:
        return self.data["players"]

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def next_id(self) -> str:
        n = 1
        while f"PLAYER_{n:02d}" in self.players:
            n += 1
        return f"PLAYER_{n:02d}"

    def ensure_player(self, pid: str, team: str | None = None, number: str | None = None,
                      source: str = "user") -> dict:
        p = self.players.get(pid)
        if p is None:
            p = {"player_id": pid, "team": team, "number": number, "samples": [],
                 "history": [{"at": now(), "action": "created", "source": source}]}
            self.players[pid] = p
        return p

    def set_attr(self, pid: str, team: str | None = None, number: str | None = None, source: str = "user"):
        p = self.ensure_player(pid)
        old = {"team": p["team"], "number": p["number"]}
        if team is not None:
            p["team"] = team or None
        if number is not None:
            p["number"] = number or None
        p["history"].append({"at": now(), "action": "set_attr", "old": old,
                             "new": {"team": p["team"], "number": p["number"]}, "source": source})

    def add_sample(self, pid: str, det_id: int, frame: int, box: list, quality: dict,
                   view: str = "unknown", source: str = "user", image: str | None = None) -> dict:
        p = self.ensure_player(pid)
        if any(s["det_id"] == det_id for s in p["samples"]):
            return p
        s = {"det_id": int(det_id), "frame": int(frame), "box": [round(float(v), 1) for v in box],
             "quality": quality, "view": view, "source": source, "image": image, "added_at": now()}
        p["samples"].append(s)
        p["history"].append({"at": now(), "action": "add_sample", "det_id": int(det_id), "source": source})
        return p

    def remove_sample(self, pid: str, det_id: int, source: str = "user"):
        p = self.players[pid]
        p["samples"] = [s for s in p["samples"] if s["det_id"] != det_id]
        p["history"].append({"at": now(), "action": "remove_sample", "det_id": int(det_id), "source": source})

    def delete_player(self, pid: str):
        self.players.pop(pid, None)


def sample_quality(pre: dict, i: int) -> dict:
    return {"height_px": float(pre["q_height"][i]), "occlusion": round(float(pre["q_occ"][i]), 3),
            "truncated": bool(pre["q_trunc"][i]), "sharpness": round(float(pre["q_sharp"][i]), 1),
            "det_score": round(float(pre["score"][i]), 3)}


def registration_problems(p: dict, min_samples: int, min_height: float) -> list[str]:
    """Reasons why a player's registration is not strong enough to confirm anyone."""
    good = [s for s in p["samples"] if s["quality"]["height_px"] >= min_height
            and s["quality"]["occlusion"] < 0.3 and not s["quality"]["truncated"]]
    probs = []
    if len(good) < min_samples:
        probs.append(f"only {len(good)} good-quality samples (need {min_samples})")
    if len({s['frame'] // 30 for s in good}) < min(2, min_samples):
        probs.append("samples are not from different moments (need >=2 distinct seconds)")
    return probs


def gallery(reg: Registry, pre: dict, min_height: float) -> dict[str, dict[str, np.ndarray]]:
    pos = {int(d): i for i, d in enumerate(pre["det_id"])}
    out = {}
    for pid, p in reg.players.items():
        idx = [pos[s["det_id"]] for s in p["samples"] if s["det_id"] in pos]
        if not idx:
            continue
        idx = np.array(idx)
        out[pid] = {k: pre[k][idx].astype(np.float32) for k in ["reid", "torso", "legs", "shoes", "hair"]}
        out[pid]["frames"] = pre["frame"][idx]
    return out
