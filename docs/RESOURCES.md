# Resources used

A complete inventory of what was used to build and validate hoopid 0.1. Versions are the exact ones installed.

## 1. People, inputs and the AI agent

| Resource | Role |
|---|---|
| Brief from the project's chief engineer (`docs/TASK_PROMPT.md`, English: `TASK_PROMPT.en.md`) | requirements, rules, metrics, deliverables |
| Test video `VIDEO-2026-10-02-14-05-11.mp4` (phone recording, 54.6 s) supplied by the user | the only data source; no external datasets were used |
| Claude (Anthropic), working as an agent in Claude Code on the web | wrote all code, ran the experiments, hand-labelled the ground truth from crop sheets, wrote the documentation, the presentation and the business plan |

## 2. Hardware and environment

| Resource | Details |
|---|---|
| Compute | Claude Code cloud container: Intel Xeon @ 2.10 GHz, 4 vCPU, 15 GB RAM, **no GPU** |
| OS | Linux 6.18 |
| Target machine (not used yet) | NVIDIA RTX 5080 16 GB, Ryzen 9950X, 96 GB RAM: speed on it is not measured |

## 3. Programming languages and runtimes

| Resource | Version | Use |
|---|---|---|
| Python | 3.11.15 | all code (`hoopid/` package, ~3,300 lines) |
| Bash | system | scripts `scripts/download_models.sh`, `scripts/run_all.sh` |
| HTML/CSS/JavaScript | — | the local web UI (served by Flask) |
| Python `venv` | — | isolated environment `.venv`, nothing installed system-wide |

## 4. Python libraries (`requirements.txt`)

| Library | Version | Licence | Used for |
|---|---|---|---|
| onnxruntime | 1.30.0 | MIT | running all neural networks (CPU provider; `onnxruntime-gpu` for RTX) |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 / MIT | video decoding, image processing, ORB features + RANSAC (camera motion), drawing, video writing |
| numpy | 2.4.6 | BSD-3 | arrays, all numerics |
| scipy | 1.17.1 | BSD-3 | Hungarian algorithm (`linear_sum_assignment`), L-BFGS-B optimiser for the logistic score model |
| Flask | 3.1.3 | BSD-3 | local registration and correction web UI |
| Pillow | 12.3.0 | MIT-CMU | drawing the coach panel text (Cyrillic/Latin fonts) |
| rapidocr-onnxruntime | 1.4.4 | Apache-2.0 | jersey-number OCR (plus its dependencies: PyYAML, Shapely, pyclipper, six) |
| pytest | 9.1.1 | MIT | unit tests (`tests/`) |

## 5. Neural network models

| Model | Source | Licence | Role |
|---|---|---|---|
| YOLOX-m (COCO, ONNX, v0.1.1rc0) | GitHub release of Megvii-BaseDetection/YOLOX | Apache-2.0 | person detection; ball detection (COCO class "sports ball") |
| YOLOX-s (ONNX) | same release | Apache-2.0 | downloaded as a faster alternative; not used in the final runs |
| YoutuReID person Re-ID (ONNX, 2021nov) | OpenCV Zoo (GitHub, LFS) | Apache-2.0 | 768-d appearance embedding, the main identity cue |
| PP-OCR detection + recognition (ONNX) | bundled in the RapidOCR wheel (PyPI) | Apache-2.0 | number reading (found no numbers in this video) |

## 6. Algorithms and methods (own implementations unless stated)

| Method | Where |
|---|---|
| ByteTrack-style two-stage association (idea from the ByteTrack paper; own code) | `tracker.py` |
| Constant-velocity Kalman filter | `tracker.py` |
| Camera-motion compensation: ORB keypoints + RANSAC similarity (OpenCV) | `camera.py` |
| Shot-cut detection by colour-histogram correlation | `camera.py` |
| HSV colour histograms with floor-pixel removal (shoes) and Bhattacharyya similarity | `features.py` |
| Joint assignment with the Hungarian algorithm (SciPy) and best-vs-second margin | `identity.py` |
| Segmenting tracks at contacts and appearance change points | `identity.py` |
| Logistic-regression evidence model fitted on the tuning half (SciPy L-BFGS-B) | `calibrate.py` |
| Grid search of thresholds on the tuning half only | `calibrate.py`, `experiments.py` |
| IDF1 metric and identity metrics against ground truth | `evaluate.py` |
| Pinhole-camera ground-position estimate (distance/speed) | `stats.py` |
| Ball-to-player attribution by proximity with ambiguity rejection | `stats.py` |

## 7. Command-line tools

| Tool | Use |
|---|---|
| ffmpeg / ffprobe | video metadata audit, exact frame timestamps, H.264 encoding of output videos, frame extraction |
| git | version control; branch `claude/happy-hopper-84jpw1` |
| curl, sha256sum | model downloads and integrity checks |
| lscpu, free, nproc | hardware audit |

## 8. Fonts

| Font | Use |
|---|---|
| DejaVu Sans (system) | text on the coach panel video |
| Google Fonts (in the presentation only) | slide typography |

## 9. Online services

| Service | Use |
|---|---|
| GitHub (repository `rybiroed/ai-vision`) | code, results and documents storage |
| GitHub Releases, raw/media.githubusercontent.com | model weight downloads |
| PyPI | Python packages |
| Claude Artifacts (claude.ai) | the presentation |
| Not reachable from the environment: Hugging Face, download.pytorch.org, Google Drive | so OSNet/TorchReID weights and PyTorch wheels were not used |

## 10. Deliberately not used

PyTorch and CUDA (not needed: ONNX Runtime), GPUs (not available), Ultralytics YOLO and BoxMOT (AGPL-3.0),
face recognition (biometrics, children in frame), super-resolution or generative restoration (forbidden as evidence),
skin-tone features (lighting-dominated, risk of a demographic proxy), external datasets.
