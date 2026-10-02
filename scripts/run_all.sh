#!/usr/bin/env bash
# Full reproduction: precompute -> baseline/system -> experiments -> correction simulation.
set -euo pipefail
cd "$(dirname "$0")/.."
V=${1:-data/raw/VIDEO-2026-10-02-14-05-11.mp4}
python -m hoopid.precompute --video "$V" --out runs/full/precompute
for m in baseline system; do
  python -m hoopid.pipeline --pre runs/full/precompute --registry match/players.json --out runs/final/$m --mode $m --video "$V"
done
python -m hoopid.experiments --out runs/report
python -m hoopid.ocr --video "$V" --pre runs/full/precompute --out runs/report/ocr.json
python -m hoopid.correction_sim --run runs/final/system --out runs/final/system_corrected
for s in 0,820 820,1638; do python -m hoopid.evaluate --tracks runs/final/system/tracks.csv --frames $s; done
