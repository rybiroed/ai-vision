"""Command-line registration (the web UI in ui.py does the same through forms).

  python -m hoopid.register --pre runs/full/precompute --registry match/players.json \
      add --player PLAYER_01 --team white --number "" --det 1234 --view front
  ... add --player PLAYER_01 --frame 300 --x 120 --y 400      # pick the box under a point
  ... remove --player PLAYER_01 --det 1234
  ... attr --player PLAYER_01 --team white --number 7
  ... list
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from .precompute import load
from .registry import Registry, registration_problems, sample_quality
from .review import collect_crops


def det_at(pre: dict, frame: int, x: float, y: float) -> int | None:
    idx = np.where(pre["frame"] == frame)[0]
    best, area = None, None
    for i in idx:
        x1, y1, x2, y2 = pre["box"][i]
        if x1 <= x <= x2 and y1 <= y <= y2:
            a = (x2 - x1) * (y2 - y1)
            if area is None or a < area:
                best, area = int(i), a
    return best


def add_sample(reg: Registry, pre: dict, video: str | None, pid: str, i: int, view: str,
               source: str, team=None, number=None, crops_dir: str | None = None) -> dict:
    reg.ensure_player(pid, source=source)
    if team is not None or number is not None:
        reg.set_attr(pid, team, number, source)
    img_path = None
    if video and crops_dir:
        os.makedirs(crops_dir, exist_ok=True)
        c = collect_crops(video, pre, [i])[i]
        img_path = os.path.join(crops_dir, f"{pid}_{int(pre['det_id'][i])}.jpg")
        cv2.imwrite(img_path, c)
    return reg.add_sample(pid, int(pre["det_id"][i]), int(pre["frame"][i]), pre["box"][i].tolist(),
                          sample_quality(pre, i), view, source, img_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--source", default="user:cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--player", required=True)
    a.add_argument("--det", type=int, action="append", default=[])
    a.add_argument("--frame", type=int); a.add_argument("--x", type=float); a.add_argument("--y", type=float)
    a.add_argument("--team"); a.add_argument("--number"); a.add_argument("--view", default="unknown")
    r = sub.add_parser("remove"); r.add_argument("--player", required=True); r.add_argument("--det", type=int, required=True)
    t = sub.add_parser("attr"); t.add_argument("--player", required=True); t.add_argument("--team"); t.add_argument("--number")
    sub.add_parser("list")
    args = ap.parse_args()
    pre = load(args.pre)
    reg = Registry(args.registry)
    pos = {int(d): k for k, d in enumerate(pre["det_id"])}
    crops_dir = os.path.join(os.path.dirname(args.registry), "samples")
    if args.cmd == "add":
        dets = list(args.det)
        if args.frame is not None:
            i = det_at(pre, args.frame, args.x, args.y)
            if i is None:
                raise SystemExit("no detection under that point")
            dets.append(int(pre["det_id"][i]))
        for d in dets:
            add_sample(reg, pre, args.video, args.player, pos[d], args.view, args.source,
                       args.team, args.number, crops_dir)
        reg.save()
    elif args.cmd == "remove":
        reg.remove_sample(args.player, args.det, args.source); reg.save()
    elif args.cmd == "attr":
        reg.set_attr(args.player, args.team, args.number, args.source); reg.save()
    for pid, p in reg.players.items():
        print(pid, p["team"], p["number"], len(p["samples"]), "samples",
              registration_problems(p, 2, 70) or "OK")


if __name__ == "__main__":
    main()
