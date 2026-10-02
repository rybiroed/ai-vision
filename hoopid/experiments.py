"""Comparison and ablation experiments.

For every variant the evidence model and thresholds are fitted on the TUNING
split only; the TEST split is evaluated once with the frozen parameters and
never used for any choice.

    python -m hoopid.experiments --out runs/report
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

from .calibrate import fit_evidence, tune
from .config import load_config
from .evaluate import evaluate, load_gt, load_pred
from .identity import Calibrator
from .pipeline import run as run_pipeline
from .precompute import load
from .registry import Registry, gallery
from .tracker import run_tracker

GRID = {"abs_thr": [-1.0, 0.0, 1.0], "margin": [0.5, 1.0, 2.0], "n_min": [2.0, 4.0],
        "col_margin": [0.0, 0.5], "contact_extra_margin": [0.5, 1.0, 2.0]}
FEATURE_SETS = {
    "reid": ["reid"],
    "colour_only": ["torso", "legs", "shoes"],
    "reid+body_colour": ["reid", "torso", "legs"],
    "reid+body_colour+shoes": ["reid", "torso", "legs", "shoes"],
    "all(+hair)": ["reid", "torso", "legs", "shoes", "hair"],
}
KEYS = ["correct_confirmed", "wrong_assignments", "wrong_on_registered", "wrong_on_other_people",
        "precision_confirmed", "coverage_correct", "unknown_share", "id_switches", "track_switches",
        "impure_tracks", "idf1_player", "idf1_track", "recovered_within_3s", "labelled_dets", "target_dets"]


def summarize(r):
    return {k: r[k] for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="runs/full/precompute")
    ap.add_argument("--registry", default="match/players.json")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--gt", default="gt/det_labels.csv")
    ap.add_argument("--map", default="gt/registration_map.json")
    ap.add_argument("--tune", default="0,820")
    ap.add_argument("--test", default="820,1638")
    ap.add_argument("--out", default="runs/report")
    ap.add_argument("--max-wrong-rate", type=float, default=0.001)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    pre = load(a.pre)
    reg = Registry(a.registry)
    gal = gallery(reg, pre, cfg["identity"]["reg_min_height"])
    reg_map = json.load(open(a.map))
    rows = list(csv.DictReader(open(a.gt)))
    gt = load_gt(a.gt)
    tune_fr = tuple(int(x) for x in a.tune.split(","))
    test_fr = tuple(int(x) for x in a.test.split(","))
    tune_rows = [r for r in rows if tune_fr[0] <= int(r["frame"]) < tune_fr[1]]
    sample_dets = {s["det_id"] for p in reg.players.values() for s in p["samples"]}
    sample_units = {int(r["unit"]) for r in rows if int(r["det_id"]) in sample_dets}
    tr = run_tracker(pre, cfg["tracker"])
    results = {}

    def ev(name, out_dir, extra=None):
        pred = load_pred(os.path.join(out_dir, "tracks.csv"))
        res = {}
        for split, fr in [("tune", tune_fr), ("test", test_fr)]:
            res[split] = summarize(evaluate(pred, gt, reg_map, fr))
            res[split + "_online"] = summarize(evaluate(pred, gt, reg_map, fr, key="pid_online"))
            full = evaluate(pred, gt, reg_map, fr)
            res[split + "_recovery"] = full["recovery"]
        res.update(extra or {})
        results[name] = res
        print(name, "tune cov", res["tune"]["coverage_correct"], "wrong", res["tune"]["wrong_assignments"],
              "| test cov", res["test"]["coverage_correct"], "wrong", res["test"]["wrong_assignments"], flush=True)

    # baseline: detector + plain IoU tracker + nearest-gallery label per track
    d = os.path.join(a.out, "baseline")
    run_pipeline(a.pre, a.registry, None, d, cfg, "baseline", pre=pre, quiet=True)
    ev("baseline", d)

    calibs = {}
    for name, feats in FEATURE_SETS.items():
        t0 = time.time()
        params = fit_evidence(pre, gal, tune_rows, reg_map, feats, sample_units, cfg)
        best, _ = tune(pre, tr, gal, cfg, Calibrator(params), gt, reg_map, tune_fr, GRID, a.max_wrong_rate, feats)
        params["thresholds"] = {k: v for k, v in (best[1] if best else {}).items() if k in GRID}
        params["tuning"] = {"frames": list(tune_fr), "max_wrong_rate": a.max_wrong_rate, "feasible": best is not None}
        cp = os.path.join(a.out, "calib", f"{name}.json")
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        json.dump(params, open(cp, "w"), indent=2)
        calibs[name] = cp
        d = os.path.join(a.out, "features", name)
        run_pipeline(a.pre, a.registry, None, d, cfg, "system", features=feats, calib_path=cp, pre=pre, quiet=True)
        ev(f"system[{name}]", d, {"calibration": cp, "thresholds": params["thresholds"],
                                  "fit_s": round(time.time() - t0, 1)})

    # choose the default feature set on the TUNING split only
    def tune_score(n):
        r = results[f"system[{n}]"]["tune"]
        return (r["wrong_assignments"], -r["coverage_correct"])
    chosen = min(FEATURE_SETS, key=tune_score)
    results["chosen_feature_set"] = chosen
    cp = calibs[chosen]

    # component ablations on the chosen set (same frozen evidence model + thresholds)
    abl = {
        "no_appearance_split": {"identity": {"appearance_split": False}},
        "no_motion_priors": {"identity": {"continuity_bonus": 0.0, "reentry_bonus": 0.0}},
        "no_tracker_appearance_gate": {"tracker": {"appearance_gate": None}},
        "no_contact_segmentation": {"identity": {"contact_iou": 1.01}},
    }
    for name, delta in abl.items():
        c2 = copy.deepcopy(cfg)
        for sec, kv in delta.items():
            c2[sec].update(kv)
        d = os.path.join(a.out, "ablation", name)
        run_pipeline(a.pre, a.registry, None, d, c2, "system", features=FEATURE_SETS[chosen], calib_path=cp,
                     pre=pre, quiet=True)
        ev(f"ablation[{name}]", d)
    json.dump(results, open(os.path.join(a.out, "experiments.json"), "w"), indent=2)
    print("chosen feature set (by tuning split):", chosen)


if __name__ == "__main__":
    main()
