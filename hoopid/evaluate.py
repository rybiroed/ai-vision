"""Evaluation against the manual ground truth (gt/det_labels.csv).

Ground truth is per detection: det_id -> identity label. Labels:
  <name>   a person on the court (e.g. "rtn_kid"); registered players map to
           these names through gt/registration_map.json
  OTHER    any other person (spectator, people on other courts, coaches)
  AMBIG    box covers several people / cannot be judged; excluded
Detections that were never labelled are excluded as well.

All metrics are computed only on labelled detections in the requested frame
range (tuning or test split). Because GT boxes are the detector's boxes,
metrics measure identity quality *given detection*; missed detections are not
counted (documented limitation). This is also why HOTA is not reported.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_gt(path: str) -> dict[int, str]:
    with open(path) as fh:
        return {int(r["det_id"]): r["gt"] for r in csv.DictReader(fh)}


def load_pred(tracks_csv: str) -> dict[int, dict]:
    out = {}
    with open(tracks_csv) as fh:
        for r in csv.DictReader(fh):
            if r["observed"] != "1":
                continue
            out[int(r["det_id"])] = {"frame": int(r["frame"]), "track_id": int(r["track_id"]),
                                     "pid": r["player_id"] if r["status"] == "CONFIRMED" else None,
                                     "status": r["status"],
                                     "pid_online": r["player_id_online"] if r["status_online"] == "CONFIRMED" else None}
    return out


def idf1(pairs: list[tuple[str, str | None]], extra_pred: int = 0) -> float:
    """pairs: (gt_id, pred_id or None) per GT detection of a target identity.
    extra_pred: predicted IDs placed on non-target people (pure false positives)."""
    gts = sorted({g for g, _ in pairs}); prs = sorted({p for _, p in pairs if p})
    if not gts:
        return float("nan")
    n_gt = len(pairs); n_pr = sum(1 for _, p in pairs if p) + extra_pred
    if not prs:
        return 0.0
    C = np.zeros((len(gts), len(prs)))
    gi = {g: k for k, g in enumerate(gts)}; pi = {p: k for k, p in enumerate(prs)}
    for g, p in pairs:
        if p:
            C[gi[g], pi[p]] += 1
    r, c = linear_sum_assignment(-C)
    idtp = C[r, c].sum()
    return float(2 * idtp / (n_gt + n_pr))


def evaluate(pred: dict, gt: dict, reg_map: dict[str, str], frames: tuple[int, int],
             key: str = "pid", fps: float = 30.0, gap_frames: int = 15) -> dict:
    """reg_map: player_id -> gt name."""
    target = set(reg_map.values())
    inv = {v: k for k, v in reg_map.items()}
    lo, hi = frames
    rows = [(d, gt[d], pred.get(d)) for d in gt
            if gt[d] != "AMBIG" and d in pred and lo <= pred[d]["frame"] < hi]
    rows.sort(key=lambda r: r[2]["frame"])
    tgt = [r for r in rows if r[1] in target]
    oth = [r for r in rows if r[1] not in target]
    correct = sum(1 for _, g, p in tgt if p[key] and reg_map.get(p[key]) == g)
    wrong_t = sum(1 for _, g, p in tgt if p[key] and reg_map.get(p[key]) != g)
    wrong_o = sum(1 for _, g, p in oth if p[key])
    confirmed = sum(1 for _, _, p in rows if p[key])
    unknown_t = sum(1 for _, _, p in tgt if not p[key])
    # identity switches per GT identity (ignoring UNKNOWN gaps)
    seq = defaultdict(list)
    for _, g, p in tgt:
        seq[g].append((p["frame"], p[key], p["track_id"]))
    id_sw, trk_sw = 0, 0
    for g, s in seq.items():
        labels = [x[1] for x in s if x[1]]
        id_sw += sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        trk_sw += sum(1 for a, b in zip(s, s[1:]) if a[2] != b[2])
    # track purity: tracks covering >1 GT identity (swap inside a track)
    tr_ids = defaultdict(Counter)
    for _, g, p in rows:
        tr_ids[p["track_id"]][g] += 1
    impure = sum(1 for c in tr_ids.values() if len(c) > 1)
    # recovery after disappearance
    rec = []
    for g, s in seq.items():
        for (f0, _, _), k in zip(s, range(len(s))):
            if k == 0:
                continue
            gap = s[k][0] - s[k - 1][0]
            if gap < gap_frames:
                continue
            f_back = s[k][0]
            after = [x for x in s[k:] if x[0] - f_back <= 3 * fps]
            first_ok = next((x[0] for x in after if x[1] and reg_map.get(x[1]) == g), None)
            wrong_after = sum(1 for x in after if x[1] and reg_map.get(x[1]) != g)
            rec.append({"gt": g, "player_id": inv.get(g), "gone_frames": int(gap), "back_at": int(f_back),
                        "recovered_s": None if first_ok is None else round((first_ok - f_back) / fps, 2),
                        "wrong_dets_in_3s": wrong_after})
    pairs = [(g, p[key]) for _, g, p in tgt]
    trk_pairs = [(g, str(p["track_id"])) for _, g, p in tgt]
    n_t = max(len(tgt), 1)
    return {
        "frames": [lo, hi], "labelled_dets": len(rows), "target_dets": len(tgt), "other_dets": len(oth),
        "correct_confirmed": correct,
        "wrong_assignments": wrong_t + wrong_o, "wrong_on_registered": wrong_t, "wrong_on_other_people": wrong_o,
        "precision_confirmed": round(correct / confirmed, 4) if confirmed else None,
        "coverage_correct": round(correct / n_t, 4),
        "unknown_share": round(unknown_t / n_t, 4),
        "id_switches": id_sw, "track_switches": trk_sw, "impure_tracks": impure,
        "idf1_player": round(idf1(pairs, wrong_o), 4),
        "idf1_track": round(idf1(trk_pairs), 4),
        "recovery": rec,
        "recovered_within_3s": f"{sum(1 for r in rec if r['recovered_s'] is not None)}/{len(rec)}",
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--gt", default="gt/det_labels.csv")
    ap.add_argument("--map", default="gt/registration_map.json")
    ap.add_argument("--frames", default="0,100000")
    ap.add_argument("--online", action="store_true", help="score the online (first-pass) decisions")
    a = ap.parse_args()
    with open(a.map) as fh:
        m = json.load(fh)
    lo, hi = (int(x) for x in a.frames.split(","))
    r = evaluate(load_pred(a.tracks), load_gt(a.gt), m, (lo, hi), "pid_online" if a.online else "pid")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
