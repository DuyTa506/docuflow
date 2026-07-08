#!/usr/bin/env bash
# Health check: infra + llama + vLLM OCR + API + Temporal worker.
#
# Usage:
#   bash deploy/check-backend.sh           # host API (systemd)
#   bash deploy/check-backend.sh --docker  # API/worker in Docker
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=1; }

DOCKER_MODE=0
[[ "${1:-}" == "--docker" ]] && DOCKER_MODE=1

FAIL=0
LLAMA_CONTAINER="llamacpp-qwen3.5-9b"
LLAMA_HOST_PORT=5011
VLLM_PORT=8000
TEMPORAL_PORT=7233
API_PORT=8022
if [[ -f "$ROOT/.env" ]]; then
  val=$(grep -E '^API_PORT=' "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
  [[ -n "$val" ]] && API_PORT="$val"
fi

echo -e "${CYAN}DocuFlow stack${NC} ($([[ "$DOCKER_MODE" -eq 1 ]] && echo "docker API" || echo "host API"))"
echo ""

# ── Infrastructure containers ────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  fail "docker not installed"
else
  for c in docuflow-postgres docuflow-minio docuflow-temporal; do
    if docker ps --format '{{.Names}}' | grep -qx "$c"; then
      ok "container '$c' running"
    elif docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
      fail "container '$c' stopped — run: sudo systemctl start docuflow-infra"
    else
      fail "container '$c' missing — run: bash scripts/start_infra.sh"
    fi
  done
fi

if bash "$ROOT/scripts/wait_for_port.sh" localhost "$TEMPORAL_PORT" 3 2>/dev/null; then
  ok "Temporal gRPC on :${TEMPORAL_PORT}"
else
  fail "Temporal not reachable on :${TEMPORAL_PORT}"
fi

# ── llama.cpp (host GPU) ─────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -qx "$LLAMA_CONTAINER"; then
  ok "llama.cpp '$LLAMA_CONTAINER' (http://localhost:${LLAMA_HOST_PORT})"
elif docker ps -a --format '{{.Names}}' | grep -qx "$LLAMA_CONTAINER"; then
  fail "llama.cpp stopped — docker start $LLAMA_CONTAINER"
else
  fail "llama.cpp missing — docker compose -f SETUPS/llms/docker-compose.yml up -d qwen3.5-9b"
fi

# ── vLLM OCR (host GPU) ─────────────────────────────────────────────
if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
  ok "DeepSeek-OCR-2 vLLM on :${VLLM_PORT}"
elif curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  ok "DeepSeek-OCR-2 vLLM responding on :${VLLM_PORT}"
else
  fail "vLLM OCR not ready on :${VLLM_PORT} — bash serve_deepseek_ocr.sh"
fi

# ── API ──────────────────────────────────────────────────────────────
if [[ "$DOCKER_MODE" -eq 1 ]]; then
  if docker ps --format '{{.Names}}' | grep -qx docuflow-api; then
    ok "docuflow-api container running"
  else
    fail "docuflow-api not running — sudo systemctl start docuflow-docker-stack"
  fi
else
  if systemctl is-active --quiet docuflow-backend.service 2>/dev/null; then
    ok "systemd docuflow-backend active"
  else
    warn "docuflow-backend not active (may still be starting)"
  fi
fi

if curl -sf "http://localhost:${API_PORT}/" >/dev/null 2>&1; then
  ok "DocuFlow API on :${API_PORT}"
else
  fail "API not responding on :${API_PORT}"
fi

# ── Temporal worker ──────────────────────────────────────────────────
if [[ "$DOCKER_MODE" -eq 1 ]]; then
  if docker ps --format '{{.Names}}' | grep -qx docuflow-temporal-worker; then
    ok "docuflow-temporal-worker container running"
  else
    fail "worker container not running"
  fi
else
  if systemctl is-active --quiet docuflow-temporal-worker.service 2>/dev/null; then
    ok "systemd docuflow-temporal-worker active"
  else
    fail "docuflow-temporal-worker not active — sudo systemctl start docuflow-temporal-worker"
  fi
fi

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  ok "All checks passed."
else
  warn "Some checks failed."
  exit 1
fi
