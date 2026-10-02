"""Stage 1: stream the video once; detect people, estimate camera motion and
extract per-detection features. Results are written in shards of
`chunk` frames so that a full match never has to sit in RAM.

Everything downstream (tracking, identity, corrections, evaluation) reads these
shards and can be re-run in seconds without touching the detector again.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import time

import numpy as np

from .camera import MotionEstimator
from .config import load_config
from .detect import YoloxDetector
from .features import ReidModel, extract
from .video import frame_timestamps, iter_frames, probe


def _flush(out_dir: str, shard: int, rows: dict) -> None:
    arrays = {k: np.concatenate(v) if v and isinstance(v[0], np.ndarray) else np.array(v)
              for k, v in rows.items()}
    np.savez_compressed(os.path.join(out_dir, f"shard_{shard:05d}.npz"), **arrays)


def run(video: str, out_dir: str, cfg: dict, end: int | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    dcfg, fcfg = cfg["detector"], cfg["features"]
    det = YoloxDetector(dcfg["model"], score_thr=dcfg["store_score_thr"],
                        nms_thr=dcfg["nms_thr"], tiles=dcfg.get("tiles", "none"))
    reid = ReidModel(fcfg["reid_model"]) if fcfg.get("reid_model") else None
    cmc = MotionEstimator(min_inliers=cfg["camera"]["min_inliers"],
                          cut_corr=cfg["camera"]["cut_hist_corr"])
    ts = frame_timestamps(video)
    meta = probe(video)
    chunk = cfg.get("chunk_frames", 600)

    det_keys = ["frame", "det_id", "box", "score", "reid", "torso", "legs", "shoes", "hair",
                "q_height", "q_trunc", "q_sharp", "q_occ"]
    cam_keys = ["cam_frame", "cam_t", "cam_A", "cam_inliers", "cam_reliable", "cam_cut", "cam_hist"]
    rows = {k: [] for k in det_keys + cam_keys}
    timing = {"detect": 0.0, "camera": 0.0, "features": 0.0}
    shard, n_det, n_frames, h_w = 0, 0, 0, None
    t0 = time.time()
    for fr in iter_frames(video, end=end, timestamps=ts):
        h_w = fr.image.shape[:2]
        a = time.time(); boxes = det(fr.image); timing["detect"] += time.time() - a
        a = time.time(); cam = cmc(fr.image, boxes); timing["camera"] += time.time() - a
        a = time.time(); f = extract(fr.image, boxes, reid); timing["features"] += time.time() - a
        n = len(boxes)
        rows["frame"].append(np.full(n, fr.idx, np.int32))
        rows["det_id"].append(np.arange(n_det, n_det + n, dtype=np.int64))
        rows["box"].append(boxes[:, :4].astype(np.float32).reshape(-1, 4))
        rows["score"].append(boxes[:, 4].astype(np.float32))
        for k in ["reid", "torso", "legs", "shoes", "hair", "q_height", "q_trunc", "q_sharp", "q_occ"]:
            rows[k].append(f[k])
        rows["cam_frame"].append(np.array([fr.idx], np.int32))
        rows["cam_t"].append(np.array([fr.t], np.float64))
        rows["cam_A"].append(cam["A"][None])
        rows["cam_inliers"].append(np.array([cam["inliers"]], np.int32))
        rows["cam_reliable"].append(np.array([cam["reliable"]], bool))
        rows["cam_cut"].append(np.array([cam["cut"]], bool))
        rows["cam_hist"].append(np.array([cam["hist_corr"]], np.float32))
        n_det += n
        n_frames += 1
        if n_frames % chunk == 0:
            _flush(out_dir, shard, rows); shard += 1
            rows = {k: [] for k in rows}
            el = time.time() - t0
            print(f"[precompute] {n_frames} frames, {n_det} dets, {n_frames / el:.2f} fps", flush=True)
    if rows["cam_frame"]:
        _flush(out_dir, shard, rows)
    wall = time.time() - t0
    stats = {
        "video": video, "frames": n_frames, "detections": n_det, "frame_hw": list(h_w or []),
        "wall_s": round(wall, 2), "fps": round(n_frames / max(wall, 1e-9), 3),
        "stage_s": {k: round(v, 2) for k, v in timing.items()},
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "providers": det.sess.get_providers(),
        "video_meta": {"codec": meta["streams"][0].get("codec_name"),
                       "coded_wh": [meta["streams"][0].get("width"), meta["streams"][0].get("height")],
                       "duration_s": float(meta["format"]["duration"])},
    }
    with open(os.path.join(out_dir, "precompute_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats


def load(out_dir: str) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "shard_*.npz"))):
        with np.load(p) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--end", type=int, default=None, help="stop after this frame (smoke tests)")
    a = ap.parse_args()
    print(json.dumps(run(a.video, a.out, load_config(a.config), a.end), indent=2))


if __name__ == "__main__":
    main()
