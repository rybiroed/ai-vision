# Models, versions, sources, licences

Checked on 2026-10-02 against LICENSE files and source pages. This is a technical summary, not
legal advice; a lawyer must review it before commercial launch. Full inventory of everything used:
`docs/RESOURCES.md`. Russian original: `docs/ru/MODELS_AND_LICENSES.ru.md`.

## Models

| Purpose | Model | Source (exact URL) | SHA-256 | Code/weights licence | Why chosen |
|---|---|---|---|---|---|
| Person detector (and ball, COCO class 32) | YOLOX-m, COCO, ONNX, 640×640 input | github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx | `21ff6cfd…3f` | Apache-2.0 | permissive licence (unlike AGPL-3.0 Ultralytics YOLOv8/11), ready ONNX, good accuracy on small people. Alternative YOLOX-s (`c5c2d13e…63`) is 2× faster on CPU; accuracy not compared |
| Appearance Re-ID | YoutuReID (2021nov), 768-d, ONNX, 128×256 input | media.githubusercontent.com/media/opencv/opencv_zoo/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx | `05796833…0d` | Apache-2.0 (models/LICENSE_youtureid) | reachable from this environment (Hugging Face and the Google Drive hosting OSNet weights were blocked), permissive licence; "same / different" AUC on tuning 0.996 |
| Number OCR | RapidOCR 1.4.4 (PP-OCR det+rec, ONNX, weights inside the wheel) | PyPI `rapidocr-onnxruntime==1.4.4` | — | Apache-2.0 | runs on CPU without external downloads |

Full hashes: `scripts/models.sha256`.

**Training-data risks.** COCO weights were trained on Flickr images under various CC licences.
Public Re-ID models are usually trained on Market-1501, DukeMTMC, MSMT17 and similar datasets, which are
research-only (DukeMTMC was withdrawn). The OpenCV Zoo card does not state YoutuReID's training data explicitly.
A commercial product should fine-tune or retrain Re-ID on its own consented data.

## Libraries (pinned in requirements.txt)

| Package | Version | Licence |
|---|---|---|
| onnxruntime | 1.30.0 | MIT |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (OpenCV), MIT (wrapper); bundles FFmpeg under LGPL |
| numpy | 2.4.6 | BSD-3 |
| scipy | 1.17.1 | BSD-3 |
| Flask | 3.1.3 | BSD-3 |
| Pillow | 12.3.0 | MIT-CMU (HPND) |
| rapidocr-onnxruntime | 1.4.4 | Apache-2.0 (pulls opencv-python and PyYAML, Shapely, pyclipper, six — MIT/BSD/Boost) |
| ffmpeg / ffprobe (system) | called as separate programs | LGPL/GPL depending on build, not linked |

## Deliberately not used

* Ultralytics YOLOv8/YOLO11 and BoxMOT: AGPL-3.0, would require open-sourcing a network service or
  buying a commercial licence.
* Generative super-resolution: forbidden by the brief as evidence of details.
* Faces: face recognition is not used at all. It is biometrics (and there are children in frame), and faces
  are not distinguishable at this distance anyway.
