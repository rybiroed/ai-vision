# hoopid: persistent basketball player IDs in video (prototype 0.1)

The user registers players as `PLAYER_01`, `PLAYER_02`, … The system follows each player,
keeps the ID through crossings and restores it after the player disappears. When in doubt it shows
`UNKNOWN` instead of guessing.

* Report with metrics, errors and limitations: **[docs/REPORT.md](docs/REPORT.md)**
* Everything used (software, models, data, services, hardware): **[docs/RESOURCES.md](docs/RESOURCES.md)**
* Models, versions, licences: [docs/MODELS_AND_LICENSES.md](docs/MODELS_AND_LICENSES.md)
* Ground truth and its change log: [gt/README.md](gt/README.md)
* Business plan: [docs/BUSINESS_PLAN.md](docs/BUSINESS_PLAN.md)
* Original brief: [docs/TASK_PROMPT.en.md](docs/TASK_PROMPT.en.md) (Russian original: [docs/TASK_PROMPT.md](docs/TASK_PROMPT.md))
* Presentation (English slides, Russian speaker notes): [docs/presentation/](docs/presentation/)
* Russian versions of all documents: [docs/ru/](docs/ru/)

## Results on the attached video (54.6 s, test half 27 s)

| | Baseline | System | System + 17 corrections |
|---|---|---|---|
| Wrong identity assignments | 3915 | **0** | 0 |
| Correctly confirmed visible time | 79.2% | **90.9%** | 95.1% |
| IDF1 | 0.619 | 0.953 | 0.975 |

Figures come from a CPU machine (4 vCPU Xeon), **not an RTX 5080**. The video is short, so independent
validation is insufficient. Details in the report.

Ready outputs are in `results/`:

| File | What it is |
|---|---|
| `results/system/annotated.mp4` | annotated video: P01…P09, `?` = UNKNOWN, `x` = unregistered, dashed red = LOST (prediction) |
| `results/system/tracks.csv` | time, coordinates, track_id, segment, player_id, status, score, margin, candidates; online and second-pass decisions |
| `results/system/assignments.jsonl` | log: track start/end, splits, assignments with reasons, revocations, second-pass revisions |
| `results/system/players.json` | player registry at run time |
| `results/baseline/*` | the same for the baseline |
| `results/system_corrected/*` | after the simulated manual correction (`corrections.json`) |
| `results/experiments.json` | comparison, feature contribution, ablations (tuning and test) |
| `results/coach/coach_dashboard.mp4` | **coach panel**: video + a side panel per player with time on screen, time with ball, touches, distance (walk/run), speed, mini-map |
| `results/coach/coach_stats.csv` | final per-player stats (see report §10: these are estimates, and why) |
| `results/ocr.json`, `results/precompute_stats.json` | number readability check; speed and memory |

## Installation

Requires Python 3.10+, ffmpeg/ffprobe on PATH, about 250 MB for models.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# GPU (RTX 5080): install onnxruntime-gpu (CUDA 12.x) instead of onnxruntime and check:
#   python -c "import onnxruntime as o; print(o.get_available_providers())"
bash scripts/download_models.sh        # downloads weights and verifies SHA-256
```

Nothing in the existing environment is replaced: everything goes into `.venv` inside the project.

## Running

```bash
V=data/raw/VIDEO-2026-10-02-14-05-11.mp4

# 1. One streaming pass: detections, camera motion, features (cached to disk in shards)
python -m hoopid.precompute --video $V --out runs/full/precompute

# 2. Player registration: web UI (click a player → PLAYER_XX, team, number, view)
python -m hoopid.ui --video $V --pre runs/full/precompute --registry match/players.json \
    --corrections match/corrections.json --run runs/full/system
# http://127.0.0.1:5000  (tabs Frames / Players / Review & correct)
#   or from the command line:
python -m hoopid.register --pre runs/full/precompute --registry match/players.json --video $V \
    add --player PLAYER_01 --frame 300 --x 120 --y 400 --view front

# 3. Tracking + identity + export (seconds; re-run after every correction)
python -m hoopid.pipeline --pre runs/full/precompute --registry match/players.json \
    --corrections match/corrections.json --out runs/full/system --mode system --video $V
python -m hoopid.pipeline ... --mode baseline        # baseline for comparison

# 4. Evaluation against ground truth, threshold tuning (tuning half only), experiments
python -m hoopid.evaluate --tracks runs/full/system/tracks.csv --frames 820,1638
python -m hoopid.calibrate        # score model + thresholds → configs/calibration.json
python -m hoopid.experiments --out runs/report

# 5. Coach panel: ball detection, per-player stats, dashboard video
python -m hoopid.ball  --video $V --out runs/full/ball.npz
python -m hoopid.stats --tracks runs/full/system/tracks.csv --ball runs/full/ball.npz --out results/coach --video $V
```

Full reproduction in one command: `bash scripts/run_all.sh` (about 12 minutes on 4 vCPU, plus ~20 minutes for the ball pass).

## How it works

`det_id` (box) → `track_id` (short trajectory) → `segment` (part of a track between
contacts, gaps and appearance jumps) → `player_id` (persistent identity).
Identity is decided per segment by joint assignment (Hungarian algorithm), with a check of
the margin over the second candidate and a ban on one ID for two simultaneously visible people.
The second pass over the recording refines earlier fragments using later frames and keeps the
original decision. Details in report §3.

Configuration: `configs/default.json` (detector, tracker, rules) and `configs/calibration.json`
(feature weights and thresholds tuned on frames 0–819).

## Layout

```
hoopid/        source code
configs/       configuration and calibration
match/         player registry of this match (players.json) and sample crops
gt/            manual ground truth
data/raw/      original video (not re-encoded) and SHA-256
results/       outputs and experiment reports
docs/          report, resources, licences, business plan, figures; docs/ru = Russian versions
scripts/       model download, full run
tests/         unit tests (pytest)
```
