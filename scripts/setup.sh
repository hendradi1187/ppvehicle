#!/usr/bin/env bash
# Bootstrap on Linux (the GPU server). Usage:
#   bash scripts/setup.sh              # CPU
#   bash scripts/setup.sh --gpu cu126  # GPU + TensorRT-capable wheel
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU=0
CUDA="cu126"
BRANCH="release/2.9"
SKIP_MODELS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=1; [[ "${2:-}" == cu* ]] && { CUDA="$2"; shift; } ;;
    --branch) BRANCH="$2"; shift ;;
    --skip-models) SKIP_MODELS=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "[1/5] virtualenv"
[[ -d .venv ]] || python3 -m venv .venv
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

echo "[2/5] vendoring PaddleDetection ($BRANCH)"
if [[ ! -f vendor/PaddleDetection/deploy/pipeline/pipeline.py ]]; then
  rm -rf vendor/PaddleDetection
  git clone --depth 1 --branch "$BRANCH" --single-branch \
    https://github.com/PaddlePaddle/PaddleDetection.git vendor/PaddleDetection
else
  echo "  already present"
fi

echo "[3/5] dependencies"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e .
if [[ $GPU -eq 1 ]]; then
  echo "  switching to paddlepaddle-gpu ($CUDA)"
  "$PY" -m pip uninstall -y paddlepaddle
  "$PY" -m pip install paddlepaddle-gpu==3.3.1 \
    -i "https://www.paddlepaddle.org.cn/packages/stable/$CUDA/"
fi

if [[ $SKIP_MODELS -eq 0 ]]; then
  echo "[4/5] models (~282 MB)"
  "$PY" -m lalin models --download
else
  echo "[4/5] skipping models"
fi

echo "[5/5] doctor"
"$PY" -m lalin doctor

cat <<'EOF'

Done. Next:
  source .venv/bin/activate
  python -m lalin run --scenario tracking --source data/samples/your.mp4 --profile gpu
  python -m lalin serve --host 0.0.0.0
EOF
