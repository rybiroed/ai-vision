"""Simulated manual-correction session (operator = ground truth oracle).

Counts how many UI operations a reviewer would need to (1) remove every wrong
assignment and (2) raise correct coverage to a target by assigning UNCERTAIN
segments, largest first. Operations are exactly those of the UI and are
applied through the normal corrections path, then the identity stage is
re-run and re-evaluated.

This measures the NUMBER of operations, not human time: no real operator was
timed. The UI records `seconds_spent` per operation for real measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict

from .config import load_config
from .evaluate import evaluate, load_gt, load_pred
from .pipeline import run as run_pipeline


def segments(tracks_csv: str) -> dict:
    seg = defaultdict(list)
    for r in csv.DictReader(open(tracks_csv)):
        if r["observed"] == "1" and r["track_id"] != "-1":
            seg[(int(r["track_id"]), int(r["segment"]))].append(r)
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/final/system")
    ap.add_argument("--pre", default="runs/full/precompute")
    ap.add_argument("--registry", default="match/players.json")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--gt", default="gt/det_labels.csv")
    ap.add_argument("--map", default="gt/registration_map.json")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--out", default="runs/final/system_corrected")
    a = ap.parse_args()
    gt = load_gt(a.gt)
    reg_map = json.load(open(a.map))
    inv = {v: k for k, v in reg_map.items()}
    seg = segments(os.path.join(a.run, "tracks.csv"))
    ops = []
    # 1) wrong assignments -> mark UNKNOWN (or assign the right player if the segment is one registered person)
    for (t, s), rr in seg.items():
        labs = Counter(gt.get(int(r["det_id"]), "UNLABELLED") for r in rr)
        wrong = [r for r in rr if r["status"] == "CONFIRMED" and gt.get(int(r["det_id"])) not in (None, "AMBIG")
                 and reg_map[r["player_id"]] != gt[int(r["det_id"])]]
        wrong_on = [r for r in rr if r["status_online"] == "CONFIRMED" and gt.get(int(r["det_id"])) not in (None, "AMBIG")
                    and reg_map[r["player_id_online"]] != gt[int(r["det_id"])]]
        if not wrong and not wrong_on:
            continue
        top, n = labs.most_common(1)[0]
        f0, f1 = int(rr[0]["frame"]), int(rr[-1]["frame"])
        if top in inv and n == len(rr):
            ops.append({"op": "assign", "track_id": t, "frames": [f0, f1], "player_id": inv[top], "why": "fix_wrong"})
        else:
            ops.append({"op": "unknown", "track_id": t, "frames": [f0, f1], "why": "fix_wrong"})
    # 2) coverage: assign uncertain segments that are purely one registered player, largest first
    target_total = sum(1 for d, g in gt.items() if g in inv)
    pred = load_pred(os.path.join(a.run, "tracks.csv"))
    correct = sum(1 for d, p in pred.items() if p["pid"] and gt.get(d) in inv and reg_map[p["pid"]] == gt[d])
    cands = []
    for (t, s), rr in seg.items():
        if rr[0]["status"] == "CONFIRMED":
            continue
        labs = Counter(gt.get(int(r["det_id"]), "UNLABELLED") for r in rr)
        top, n = labs.most_common(1)[0]
        if top in inv and n == len(rr):
            cands.append((len(rr), t, int(rr[0]["frame"]), int(rr[-1]["frame"]), inv[top]))
    cov_ops = 0
    for n, t, f0, f1, pid in sorted(cands, reverse=True):
        if correct / target_total >= a.target:
            break
        ops.append({"op": "assign", "track_id": t, "frames": [f0, f1], "player_id": pid, "why": "raise_coverage"})
        correct += n; cov_ops += 1
    os.makedirs(a.out, exist_ok=True)
    cp = os.path.join(a.out, "corrections.json")
    json.dump(ops, open(cp, "w"), indent=2)
    t0 = time.time()
    run_pipeline(a.pre, a.registry, cp, a.out, load_config(a.config), "system", quiet=True)
    rerun_s = time.time() - t0
    after = load_pred(os.path.join(a.out, "tracks.csv"))
    res = {"ops_total": len(ops), "ops_fix_wrong": sum(o["why"] == "fix_wrong" for o in ops),
           "ops_raise_coverage": cov_ops, "rerun_identity_s": round(rerun_s, 2),
           "video_seconds": 54.6,
           "after": {sp: {k: v for k, v in evaluate(after, gt, reg_map, fr).items() if k != "recovery"}
                     for sp, fr in [("all", (0, 10 ** 6)), ("tune", (0, 820)), ("test", (820, 1638))]}}
    json.dump(res, open(os.path.join(a.out, "correction_summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
