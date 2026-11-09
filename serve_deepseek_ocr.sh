#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/conda_env.sh"
activate_docuflow_conda

# shellcheck disable=SC1091
source "$ROOT/scripts/vllm_ocr_config.sh"
vllm_ocr_build_serve_args

exec vllm serve "${VLLM_OCR_SERVE_ARGS[@]}"
