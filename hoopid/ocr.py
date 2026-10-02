"""Jersey-number reading (RapidOCR, Apache-2.0, PP-OCR models shipped in the wheel).

Reads only the torso band of original-resolution crops (no upscaling beyond
what the OCR model's own resize does, and no generative restoration). Every
read keeps its text and confidence. A number is considered *confirmed* for a
segment only if at least `min_votes` distinct frames agree with confidence
>= `min_conf`; a single weak read never confirms anything.

    python -m hoopid.ocr --video ... --pre ... --out runs/report/ocr.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

from .precompute import load
from .review import collect_crops

MIN_TORSO_PX = 40  # torso band height below which reading is not attempted


def read_numbers(engine, crop: np.ndarray) -> list[tuple[str, float, str]]:
    h = crop.shape[0]
    torso = crop[int(0.15 * h):int(0.55 * h)]
    if torso.shape[0] < MIN_TORSO_PX:
        return []
    res, _ = engine(torso)
    out = []
    for _, text, conf in res or []:
        digits = re.sub(r"\D", "", text)
        out.append((digits, float(conf), text))
    return out


def aggregate(reads: list[tuple[int, str, float]], min_conf=0.8, min_votes=3) -> dict:
    votes = Counter(d for f, d, c in reads if d and c >= min_conf)
    frames = defaultdict(set)
    for f, d, c in reads:
        if d and c >= min_conf:
            frames[d].add(f)
    cands = sorted(((len(frames[d]), d) for d in votes), reverse=True)
    confirmed = cands[0][1] if cands and cands[0][0] >= min_votes and (len(cands) == 1 or cands[0][0] >= 2 * cands[1][0]) else None
    return {"confirmed": confirmed, "candidates": [[d, n] for n, d in cands[:5]]}


def main():
    from rapidocr_onnxruntime import RapidOCR
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--gt", default="gt/det_labels.csv")
    ap.add_argument("--per-identity", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pre = load(a.pre)
    pos = {int(d): k for k, d in enumerate(pre["det_id"])}
    rows = [r for r in csv.DictReader(open(a.gt)) if r["gt"] not in ("AMBIG", "OTHER")]
    by = defaultdict(list)
    for r in rows:
        by[r["gt"]].append(pos[int(r["det_id"])])
    pick = {}
    for g, ids in by.items():
        ids = sorted(ids, key=lambda i: -pre["q_height"][i])[:a.per_identity]  # the largest views
        pick[g] = ids
    crops = collect_crops(a.video, pre, [i for v in pick.values() for i in v], pad=0.0)
    eng = RapidOCR()
    report = {"min_torso_px": MIN_TORSO_PX, "identities": {}}
    for g, ids in pick.items():
        reads, texts, attempted = [], Counter(), 0
        for i in ids:
            r = read_numbers(eng, crops[i])
            if crops[i].shape[0] * 0.4 >= MIN_TORSO_PX:
                attempted += 1
            for d, c, t in r:
                texts[t] += 1
                reads.append((int(pre["frame"][i]), d, c))
        report["identities"][g] = {"crops": len(ids), "attempted": attempted,
                                   "max_box_height_px": float(max(pre["q_height"][ids])),
                                   "any_text_reads": sum(texts.values()),
                                   "digit_reads": sum(1 for _, d, _ in reads if d),
                                   "top_texts": texts.most_common(5), "number": aggregate(reads)}
    json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
