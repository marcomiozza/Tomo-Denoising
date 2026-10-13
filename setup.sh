#!/bin/bash
# Sets up the project environment.
#
#   1. creates .venv with all Python dependencies (pipeline, TV/BL, U-Net)
#   2. builds the BM3D binary in methods/bm3d/build/
#
# BM3D needs OpenCV with the xphoto module, which the OpenCV pip packages do
# not include. On Ubuntu/Debian:
#   sudo apt install build-essential cmake libopencv-dev libopencv-contrib-dev
#
# Usage:
#   ./setup.sh            # CPU torch
#   ./setup.sh --gpu      # torch with CUDA support
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== [1/2] Python environment (.venv) ==="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
if [ "${1:-}" = "--gpu" ]; then
    .venv/bin/pip install torch -q
else
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi
.venv/bin/pip install -r requirements.txt -q
echo "  ok: $(.venv/bin/python -c 'import torch, skimage, pydicom; print("torch", torch.__version__, "| scikit-image", skimage.__version__, "| pydicom", pydicom.__version__)')"

echo ""
echo "=== [2/2] BM3D binary ==="
if cmake -S methods/bm3d -B methods/bm3d/build -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null \
   && cmake --build methods/bm3d/build --parallel > /dev/null; then
    echo "  ok: methods/bm3d/build/bm3d"
else
    echo "  [WARNING] BM3D build failed: OpenCV with xphoto is required (see above)."
    echo "  TV, BL and U-Net work anyway."
fi
