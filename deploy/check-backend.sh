#!/usr/bin/env bash
# Health check: llama.cpp (docker) + DeepSeek OCR (vLLM) + DocuFlow API.
#
# Usage: bash deploy/check-backend.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=1; }

FAIL=0
LLAMA_CONTAINER="llamacpp-qwen3.5-9b"
LLAMA_HOST_PORT=5011
VLLM_PORT=8000
API_PORT=8002

echo -e "${CYAN}DocuFlow backend stack${NC}"
echo "  (1) llama.cpp pipeline LLM  — docker :${LLAMA_HOST_PORT}"
echo "  (2) DeepSeek-OCR-2          — vLLM   :${VLLM_PORT}  (serve_deepseek_ocr.sh / start.sh)"
echo "  (3) DocuFlow API            — uvicorn :${API_PORT}"
echo ""

# ── Docker / llama.cpp ───────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  fail "docker not installed"
else
  if docker ps --format '{{.Names}}' | grep -qx "$LLAMA_CONTAINER"; then
    ok "llama.cpp container '$LLAMA_CONTAINER' running (http://localhost:${LLAMA_HOST_PORT})"
  elif docker ps -a --format '{{.Names}}' | grep -qx "$LLAMA_CONTAINER"; then
    fail "container '$LLAMA_CONTAINER' exists but stopped"
    echo "       → docker start $LLAMA_CONTAINER"
    echo "       → or: docker compose -f $ROOT/SETUPS/llms/docker-compose.yml up -d qwen3.5-9b"
  else
    fail "container '$LLAMA_CONTAINER' not found"
    echo "       → docker compose -f $ROOT/SETUPS/llms/docker-compose.yml up -d qwen3.5-9b"
  fi
fi

# ── vLLM DeepSeek OCR ───────────────────────────────────────────────
if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
  ok "DeepSeek-OCR-2 vLLM healthy on :${VLLM_PORT}"
elif curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  ok "DeepSeek-OCR-2 vLLM responding on :${VLLM_PORT} (/v1/models)"
else
  fail "DeepSeek-OCR-2 vLLM not ready on :${VLLM_PORT}"
  echo "       Model load takes several minutes after boot."
  echo "       → bash $ROOT/serve_deepseek_ocr.sh"
  echo "       → or: sudo systemctl start docuflow-backend  (runs start.sh)"
  echo "       → log: tail -f $ROOT/.vllm_ocr.log"
  if [[ -f "$ROOT/.vllm_ocr.log" ]]; then
  echo "       last log lines:"
  tail -3 "$ROOT/.vllm_ocr.log" | sed 's/^/         /'
  fi
fi

# ── API ──────────────────────────────────────────────────────────────
if curl -sf "http://localhost:${API_PORT}/" >/dev/null 2>&1; then
  ok "DocuFlow API on :${API_PORT}"
else
  fail "DocuFlow API not responding on :${API_PORT}"
  echo "       → sudo systemctl start docuflow-backend"
  echo "       → journalctl -u docuflow-backend -f"
fi

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  ok "All backend services look healthy."
else
  warn "Some checks failed — OCR/translate/summarize need (1)+(2); API UI needs (3)."
  exit 1
fi
