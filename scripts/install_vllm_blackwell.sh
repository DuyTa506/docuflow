#!/usr/bin/env bash
# Install / upgrade vLLM 0.23+ for NVIDIA Blackwell (sm_120) in conda env vllm-blackwell.
# Requires: conda, uv (pip install uv), network access.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/conda_env.sh"
activate_docuflow_conda

VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
CUDA_TAG="${VLLM_CUDA_TAG:-cu129}"

echo "[install] conda env: ${CONDA_DEFAULT_ENV:-?}"
echo "[install] target: vllm==${VLLM_VERSION} (${CUDA_TAG} wheels)"

if ! command -v uv >/dev/null 2>&1; then
  echo "[install] uv not found — install with: pip install uv" >&2
  exit 1
fi

# Stop any running OCR server in this env
if pgrep -f "vllm serve.*DeepSeek-OCR" >/dev/null 2>&1; then
  echo "[install] stopping existing vLLM OCR process…"
  pkill -f "vllm serve.*DeepSeek-OCR" || true
  sleep 2
fi

uv pip uninstall vllm 2>/dev/null || true

uv pip install "vllm==${VLLM_VERSION}" \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_TAG}" \
  --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --index-strategy unsafe-best-match

# pandas 2.1.x breaks with numpy 2.3+ pulled by vLLM 0.23
uv pip install "pandas==2.2.3"

echo "[install] Blackwell sm_120 notes:"
echo "  - OCR serve sets VLLM_USE_FLASHINFER_SAMPLER=0 (system nvcc 12.8)"
echo "  - OCR serve uses --moe-backend triton (FlashInfer MoE JIT needs nvcc >= 12.9)"
echo "  - After installing CUDA toolkit >= 12.9, you can try enabling FlashInfer paths again"

python - <<'PY'
import torch, vllm
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
print(f"vllm {vllm.__version__}")
print(f"GPU cap {torch.cuda.get_device_capability()}")
print("NGramPerReqLogitsProcessor OK")
PY

echo "[install] done. Start OCR: bash serve_deepseek_ocr.sh"
