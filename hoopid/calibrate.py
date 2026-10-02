"""Fit the evidence model and tune decision thresholds on the TUNING split only.

1. Evidence model: logistic regression on (observation, registered player)
   pairs from labelled tuning detections, y = same identity. Each enabled
   feature contributes w_f * (s_f - c_f) only when it is available, so the
   weight of a feature is learned from data, and unavailable features (small
   heads, truncated feet, ...) simply drop out. Observations from the units
   used as registration samples are excluded (they would match trivially).
   The output is an uncalibrated *score* (logit scale), not a probability.
2. Thresholds: grid search on the tuning split; objective = maximise correct
   coverage subject to wrong assignments <= max_wrong_rate of labelled
   detections. The test split is never looked at here.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json

import numpy as np
from scipy.optimize import minimize

from .config import load_config
from .evaluate import evaluate
from .identity import Calibrator, IdentityManager, quality, similarities
from .precompute import load
from .registry import Registry, gallery
from .tracker import run_tracker


def fit_evidence(pre, gal, gt_rows, reg_map, feats, exclude_units, cfg, l2=1e-4) -> dict:
    pos = {int(d): k for k, d in enumerate(pre["det_id"])}
    pids = list(gal)
    rows = [r for r in gt_rows if r["gt"] != "AMBIG" and int(r["unit"]) not in exclude_units]
    idx = np.array([pos[int(r["det_id"])] for r in rows])
    gts = [r["gt"] for r in rows]
    sims = similarities(pre, idx, gal, feats)
    q = quality(pre, idx, cfg["identity"])
    keep = q > 0
    X, Y, W = [], [], []
    for j, p in enumerate(pids):
        y = np.array([g == reg_map[p] for g in gts], np.float64)
        cols = []
        for f in feats:
            s = sims[f][:, j]
            a = ~np.isnan(s)
            cols += [np.where(a, s, 0.0), a.astype(np.float64)]
        X.append(np.stack(cols, 1)[keep]); Y.append(y[keep]); W.append(q[keep])
    X = np.concatenate(X); Y = np.concatenate(Y); W = np.concatenate(W)
    # balance classes
    wpos = W * Y / max((W * Y).sum(), 1e-9); wneg = W * (1 - Y) / max((W * (1 - Y)).sum(), 1e-9)
    Wb = (wpos + wneg) * len(Y) / 2

    def loss(th):
        z = th[0] + X @ th[1:]
        l = np.logaddexp(0, z) - Y * z
        g = (1 / (1 + np.exp(-z)) - Y) * Wb
        return (Wb * l).sum() / len(Y) + l2 * (th[1:] ** 2).sum(), \
            np.concatenate([[g.sum()], X.T @ g]) / len(Y) + np.concatenate([[0], 2 * l2 * th[1:]])

    res = minimize(loss, np.zeros(1 + X.shape[1]), jac=True, method="L-BFGS-B")
    th = res.x
    params = {"bias": float(th[0]), "w": {}, "c": {}, "fit": {"n_pairs": int(len(Y)), "pos": int(Y.sum()),
                                                              "loss": float(res.fun), "features": feats}}
    for k, f in enumerate(feats):
        w, u = th[1 + 2 * k], th[2 + 2 * k]
        params["w"][f] = float(w)
        params["c"][f] = float(-u / w) if abs(w) > 1e-6 else 0.0
    return params


def tune(pre, tr, gal, cfg, calib, gt, reg_map, frames, grid, max_wrong_rate, feats, corr=None):
    from .evaluate import load_gt  # noqa
    best, table = None, []
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        c = dict(cfg["identity"]); c.update(dict(zip(keys, vals)))
        im = IdentityManager(pre, tr, gal, c, calib, corr or [], feats)
        on = im.run_online(); fin = im.run_offline(on)
        pred = _pred(pre, tr, fin, on)
        r = evaluate(pred, gt, reg_map, frames)
        ro = evaluate(pred, gt, reg_map, frames, key="pid_online")
        row = {**dict(zip(keys, vals)), "coverage": r["coverage_correct"], "wrong": r["wrong_assignments"],
               "coverage_online": ro["coverage_correct"], "wrong_online": ro["wrong_assignments"],
               "labelled": r["labelled_dets"]}
        table.append(row)
        ok = r["wrong_assignments"] <= max_wrong_rate * r["labelled_dets"] and \
            ro["wrong_assignments"] <= max_wrong_rate * r["labelled_dets"]
        score = r["coverage_correct"] + 0.5 * ro["coverage_correct"]
        if ok and (best is None or score > best[0]):
            best = (score, row)
    return best, table


def _pred(pre, tr, fin, on):
    out = {}
    for i in range(len(pre["frame"])):
        f, o = fin.get(i), on.get(i)
        if f is None:
            continue
        out[int(pre["det_id"][i])] = {"frame": int(pre["frame"][i]), "track_id": int(tr["track_of"][i]),
                                      "pid": f["player_id"] if f["status"] == "CONFIRMED" else None,
                                      "status": f["status"],
                                      "pid_online": o["player_id"] if o and o["status"] == "CONFIRMED" else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="runs/full/precompute")
    ap.add_argument("--registry", default="match/players.json")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--gt", default="gt/det_labels.csv")
    ap.add_argument("--map", default="gt/registration_map.json")
    ap.add_argument("--tune-frames", default="0,820")
    ap.add_argument("--features", default="reid,torso,legs,shoes,hair")
    ap.add_argument("--out", default="configs/calibration.json")
    ap.add_argument("--max-wrong-rate", type=float, default=0.001)
    ap.add_argument("--table", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    pre = load(a.pre)
    reg = Registry(a.registry)
    gal = gallery(reg, pre, cfg["identity"]["reg_min_height"])
    with open(a.map) as fh:
        reg_map = json.load(fh)
    lo, hi = (int(x) for x in a.tune_frames.split(","))
    with open(a.gt) as fh:
        rows = list(csv.DictReader(fh))
    tune_rows = [r for r in rows if lo <= int(r["frame"]) < hi]
    sample_dets = {s["det_id"] for p in reg.players.values() for s in p["samples"]}
    sample_units = {int(r["unit"]) for r in rows if int(r["det_id"]) in sample_dets}
    feats = a.features.split(",")
    params = fit_evidence(pre, gal, tune_rows, reg_map, feats, sample_units, cfg)
    print(json.dumps(params, indent=2))
    calib = Calibrator(params)
    tr = run_tracker(pre, cfg["tracker"])
    gt = {int(r["det_id"]): r["gt"] for r in rows}
    grid = {"abs_thr": [-1.0, 0.0, 1.0], "margin": [0.5, 1.0, 2.0], "n_min": [2.0, 4.0],
            "col_margin": [0.0, 0.5], "contact_extra_margin": [0.5, 1.0, 2.0]}
    best, table = tune(pre, tr, gal, cfg, calib, gt, reg_map, (lo, hi), grid, a.max_wrong_rate, feats)
    params["thresholds"] = best[1] if best else None
    params["tuning"] = {"frames": [lo, hi], "max_wrong_rate": a.max_wrong_rate, "grid": grid,
                        "excluded_sample_units": sorted(sample_units)}
    with open(a.out, "w") as fh:
        json.dump(params, fh, indent=2)
    if a.table:
        with open(a.table, "w") as fh:
            json.dump(table, fh, indent=1)
    print("best", json.dumps(best[1] if best else None))


if __name__ == "__main__":
    main()
