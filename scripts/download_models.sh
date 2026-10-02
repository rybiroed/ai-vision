#!/usr/bin/env bash
# Downloads pinned model weights into models/ and verifies SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
dl() { [ -f "models/$1" ] || curl -fL --retry 3 -o "models/$1" "$2"; }
dl yolox_m.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
dl yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
dl person_reid_youtu_2021nov.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx
dl LICENSE_youtureid https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/person_reid_youtureid/LICENSE
cd models && sha256sum -c ../scripts/models.sha256
