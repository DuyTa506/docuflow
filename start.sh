#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

# ── Cleanup on Ctrl-C / exit ─────────────────────────────────────────
cleanup() {
    info "Shutting down…"
    # Kill vLLM OCR if we started it
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        info "Stopping vLLM OCR server (pid $VLLM_PID)…"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    # Stop llama.cpp container if we started it
    if [[ "${DOCKER_STARTED:-false}" == "true" ]]; then
        info "Stopping llama.cpp container…"
        docker stop llamacpp-qwen3.5-9b 2>/dev/null || true
    fi
    ok "All services stopped."
}
trap cleanup EXIT INT TERM

# ── 1. LLM pipeline container (llama.cpp) ────────────────────────────
info "Checking llama.cpp container…"
if docker ps --format '{{.Names}}' | grep -q '^llamacpp-qwen3.5-9b$'; then
    ok "llama.cpp container already running"
else
    info "Starting llama.cpp container…"
    docker compose -f "$ROOT/SETUPS/llms/docker-compose.yml" up -d qwen3.5-9b
    DOCKER_STARTED=true
    ok "llama.cpp container started"
fi

# ── 2. vLLM OCR server (background, conda env) ───────────────────────
info "Starting vLLM OCR server (background)…"

# Source conda so we can use conda activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-blackwell

VLLM_LOG="$ROOT/.vllm_ocr.log"
vllm serve "deepseek-ai/DeepSeek-OCR-2" \
    --disable-custom-all-reduce \
    --enforce-eager \
    --logits-processors "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor" \
    --api-key 05062001 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 8192 \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
info "vLLM OCR PID: $VLLM_PID (log: $VLLM_LOG)"

# ── Wait for vLLM to be ready ────────────────────────────────────────
info "Waiting for vLLM OCR to be ready…"
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        ok "vLLM OCR ready after ${i}s"
        break
    fi
    sleep 1
done

# ── 3. API server (.venv, foreground) ────────────────────────────────
info "Activating .venv and starting API server…"
source "$ROOT/.venv/bin/activate"

ok "All dependencies ready — starting API on http://localhost:8002"
exec uvicorn serving.workflow_api:app --host 0.0.0.0 --port 8002 --reload
