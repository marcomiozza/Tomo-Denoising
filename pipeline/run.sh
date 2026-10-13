#!/bin/bash
# Entry point of the denoising pipeline.
#
# Usage:
#   Put the case folder (with Noisy/ inside) in pipeline/input/, then:
#   pipeline/run.sh                          # single case in input/, interactive menu
#   pipeline/run.sh case01                   # picks the case in input/
#   pipeline/run.sh case01 --methods all     # non-interactive
#   Results go to pipeline/output/<case>/
#
# Uses the .venv environment created by setup.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "[ERROR] Environment not found: $PY"
    echo "Create it with: ./setup.sh"
    exit 1
fi

exec "$PY" "$HERE/denoise_pipeline.py" "$@"
