"""Stage 2: tracking + identity + exports, from precomputed shards.

    python -m hoopid.pipeline --pre runs/full/precompute --registry match/players.json \
        --corrections match/corrections.json --out runs/full/system --mode system

--mode baseline   detector + plain IoU tracker + naive nearest-gallery labelling
--mode system     appearance-gated tracker + persistent identity manager (+ 2nd pass)
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from .config import load_config
from .corrections import expand as expand_corrections
from .export import lost_player_map, render_video, write_assignments, write_tracks_csv
from .identity import Calibrator, IdentityManager, baseline_identity
from .precompute import load
from .registry import Registry, gallery, registration_problems
from .tracker import run_tracker


def run(pre_dir: str, registry: str, corrections: str | None, out: str, cfg: dict, mode: str,
        video: str | None = None, features: list[str] | None = None, second_pass: bool = True,
        calib_path: str | None = None, pre: dict | None = None, quiet: bool = False) -> dict:
    t0 = time.time()
    pre = pre if pre is not None else load(pre_dir)
    reg = Registry(registry)
    icfg = cfg["identity"]
    problems = {pid: registration_problems(p, icfg["reg_min_samples"], icfg["reg_min_height"])
                for pid, p in reg.players.items()}
    weak = [pid for pid, pr in problems.items() if pr]
    gal = gallery(reg, pre, icfg["reg_min_height"])
    for pid in weak:  # weakly registered players are never auto-confirmed
        gal.pop(pid, None)
    corr = []
    if corrections and os.path.exists(corrections):
        with open(corrections) as fh:
            corr = json.load(fh)
    tcfg = dict(cfg["tracker"])
    if mode == "baseline":
        tcfg["appearance_gate"] = None
    tr = run_tracker(pre, tcfg)
    if corr:
        t_of = tr["track_of"]
        ranges = {}
        for i in np.where(t_of >= 0)[0]:
            f = int(pre["frame"][i]); a, b = ranges.get(int(t_of[i]), (f, f))
            ranges[int(t_of[i])] = (min(a, f), max(b, f))
        corr = expand_corrections(corr, ranges)
    log, online = [], {}
    if mode == "baseline":
        final = baseline_identity(pre, tr, gal, icfg)
        online = final
    else:
        calib = Calibrator.load(calib_path or icfg.get("calibration"), icfg["default_calibration"])
        th = calib.p.get("thresholds") or {}
        icfg = {**icfg, **{k: v for k, v in th.items() if k in icfg}}
        if features is None and calib.p.get("fit"):
            features = calib.p["fit"]["features"]
        im = IdentityManager(pre, tr, gal, icfg, calib, corr, features)
        online = im.run_online()
        final = im.run_offline(online) if second_pass else online
        log = im.log
    elapsed = time.time() - t0
    os.makedirs(out, exist_ok=True)
    lmap = lost_player_map(pre, tr["track_of"], final, tr["lost"])
    write_tracks_csv(os.path.join(out, "tracks.csv"), pre, tr["track_of"], final, online, tr["lost"], lmap)
    write_assignments(os.path.join(out, "assignments.jsonl"),
                      [{"pass_": "tracker", **e} for e in tr["events"]] + log, corr)
    with open(os.path.join(out, "players.json"), "w") as fh:
        snap = dict(reg.data)
        snap["registration_problems"] = problems
        json.dump(snap, fh, indent=2, ensure_ascii=False)
    stats = {"mode": mode, "features": features or icfg["features_enabled"], "second_pass": second_pass,
             "tracks": int(len(set(tr["track_of"][tr["track_of"] >= 0].tolist()))),
             "identity_wall_s": round(elapsed, 2),
             "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
             "weak_registrations": weak}
    with open(os.path.join(out, "run_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    if video:
        render_video(video, os.path.join(out, "annotated.mp4"), pre, tr["track_of"], final, tr["lost"], lmap,
                     title=mode)
    if not quiet:
        print(json.dumps(stats, indent=2))
    return {"pre": pre, "track": tr, "final": final, "online": online, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--corrections", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--mode", choices=["baseline", "system"], default="system")
    ap.add_argument("--video", default=None, help="render annotated.mp4 from this video")
    ap.add_argument("--features", default=None, help="comma list, e.g. reid,torso,legs")
    ap.add_argument("--no-second-pass", action="store_true")
    a = ap.parse_args()
    run(a.pre, a.registry, a.corrections, a.out, load_config(a.config), a.mode, a.video,
        a.features.split(",") if a.features else None, not a.no_second_pass)


if __name__ == "__main__":
    main()
