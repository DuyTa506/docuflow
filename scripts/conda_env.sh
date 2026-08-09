#!/usr/bin/env bash
# Resolve conda base and activate the DocuFlow vLLM env (vllm-blackwell).
# Used by start.sh and serve_deepseek_ocr.sh — systemd has no login shell / conda on PATH.
#
# Optional: export CONDA_BASE=/path/to/anaconda3 before calling.

DOCUFLOW_CONDA_ENV="${DOCUFLOW_CONDA_ENV:-vllm-blackwell}"

find_conda_base() {
  local base="${CONDA_BASE:-}"
  if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
    echo "$base"
    return 0
  fi
  local candidate home="${HOME:-}"
  for candidate in \
    "${home}/anaconda3" \
    "${home}/miniconda3" \
    "/opt/conda" \
    "/usr/local/anaconda3"; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    conda info --base 2>/dev/null && return 0
  fi
  return 1
}

activate_docuflow_conda() {
  local base
  base="$(find_conda_base)" || {
    echo "[ERR] conda not found. Install Anaconda or set CONDA_BASE in systemd/docuflow-backend.service" >&2
    echo "      e.g. CONDA_BASE=$HOME/anaconda3" >&2
    return 1
  }
  # shellcheck disable=SC1091
  source "${base}/etc/profile.d/conda.sh"
  conda activate "${DOCUFLOW_CONDA_ENV}"
}
