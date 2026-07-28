#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Load .env when present (systemd also passes EnvironmentFile)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
API_PORT="${API_PORT:-8022}"

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

SHUTTING_DOWN=false
VLLM_PID=""
VLLM_LOG="$ROOT/.vllm_ocr.log"
UVICORN_PID=""

# ── Cleanup on Ctrl-C / exit ─────────────────────────────────────────
cleanup() {
    SHUTTING_DOWN=true
    info "Shutting down…"
    if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "$UVICORN_PID" 2>/dev/null; then
        info "Stopping API server (pid $UVICORN_PID)…"
        kill "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi
    # Kill vLLM OCR if we started it
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        info "Stopping vLLM OCR server (pid $VLLM_PID)…"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    # Stop llama.cpp container if we started it
    if [[ "${DOCKER_STARTED:-false}" == "true" ]]; then
        info "Stopping llama.cpp container…"
        docker stop "$LLM_CONTAINER" 2>/dev/null || true
    fi
    ok "All services stopped."
}
trap cleanup EXIT INT TERM

# Tên service/container của model LLM pipeline. Đặt thành biến vì start.sh
# từng hardcode "qwen3.5-9b" ở 5 chỗ — đổi model trong compose là backend
# crash-loop với "no such service" (đã dính khi chuyển sang Gemma 28/07).
# Ghi đè bằng LLM_COMPOSE_SERVICE / LLM_CONTAINER trong .env nếu cần.
LLM_COMPOSE_SERVICE="${LLM_COMPOSE_SERVICE:-gemma-4-26b}"
LLM_CONTAINER="${LLM_CONTAINER:-llamacpp-${LLM_COMPOSE_SERVICE}}"



# ── 1. Infrastructure (MinIO + Postgres + Temporal) ───────────────────
info "Checking infrastructure (MinIO, Postgres, Temporal)…"
if ! command -v docker >/dev/null 2>&1; then
    err "docker not found — infrastructure containers are required."
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q '^docuflow-minio$' \
   && docker ps --format '{{.Names}}' | grep -q '^docuflow-postgres$' \
   && docker ps --format '{{.Names}}' | grep -q '^docuflow-temporal$'; then
    ok "Infrastructure containers already running"
else
    info "Starting infrastructure (postgres, minio, temporal, temporal-ui)…"
    if ! bash "$ROOT/scripts/start_infra.sh"; then
        err "Failed to start infrastructure — check: docker compose logs"
        exit 1
    fi
    DOCKER_STARTED=true
fi

# ── 2. LLM pipeline container (llama.cpp) ────────────────────────────
info "Checking llama.cpp docker ($LLM_CONTAINER → host :5011)…"
if ! command -v docker >/dev/null 2>&1; then
    err "docker not found — pipeline LLM (translate/summarize) requires docker."
    err "  Install docker, then: docker compose -f SETUPS/llms/docker-compose.yml up -d $LLM_COMPOSE_SERVICE"
    exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx "$LLM_CONTAINER"; then
    ok "llama.cpp container already running"
else
    info "Starting llama.cpp container…"
    if ! docker compose -f "$ROOT/SETUPS/llms/docker-compose.yml" up -d "$LLM_COMPOSE_SERVICE"; then
        err "Failed to start $LLM_CONTAINER — check: docker ps -a && docker logs $LLM_CONTAINER"
        exit 1
    fi
    DOCKER_STARTED=true
    ok "llama.cpp container started (host http://localhost:5011)"
fi

# ── 2. vLLM OCR server (background, conda env) ───────────────────────
info "Checking vLLM OCR server on :8000…"

if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    ok "vLLM OCR already healthy on :8000 — skipping spawn"
    VLLM_PID=""
else
    info "Starting vLLM OCR server (background)…"

    # shellcheck disable=SC1091
    source "$ROOT/scripts/conda_env.sh"
    activate_docuflow_conda || exit 1

    # Optional: structure-preserving export (PDF overlay + DocLayout ONNX)
    if ! python -c "import onnxruntime" 2>/dev/null; then
        info "onnxruntime not installed — PDF overlay translation disabled until: pip install onnxruntime onnx opencv-python-headless"
    fi
    if ! command -v pandoc >/dev/null 2>&1; then
        info "pandoc not found — LaTeX→OMML in DOCX export will use python-docx fallback"
    fi

    # shellcheck disable=SC1091
    source "$ROOT/scripts/vllm_ocr_config.sh"
    vllm_ocr_build_serve_args

    VLLM_LOG="$ROOT/.vllm_ocr.log"
    vllm serve "${VLLM_OCR_SERVE_ARGS[@]}" >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    info "vLLM OCR PID: $VLLM_PID (log: $VLLM_LOG)"

    # ── Wait for vLLM to be ready ────────────────────────────────────────
    info "Waiting for DeepSeek-OCR-2 vLLM on :8000 (same as serve_deepseek_ocr.sh)…"
    VLLM_READY=false
    for i in $(seq 1 120); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            ok "vLLM OCR ready after ${i}s"
            VLLM_READY=true
            break
        fi
        sleep 1
    done
    if [[ "$VLLM_READY" != "true" ]]; then
        err "DeepSeek-OCR-2 vLLM did not become ready within 120s."
        err "  Log: $VLLM_LOG"
        err "  Manual: bash serve_deepseek_ocr.sh"
        err "  API will start but OCR/extract will fail until vLLM is healthy."
    fi
fi

# ── 3. API server (.venv, foreground) ────────────────────────────────
info "Activating .venv and starting API server…"
source "$ROOT/.venv/bin/activate"

info "Ensuring MinIO bucket exists…"
python "$ROOT/scripts/ensure_minio_bucket.py" || {
    err "MinIO bucket setup failed — check MINIO_* in .env"
    exit 1
}

info "Initializing database schema…"
python "$ROOT/scripts/init_db.py" || {
    err "Database init failed — check DATABASE_URL in .env"
    exit 1
}

ok "All dependencies ready — starting API on http://localhost:${API_PORT}"
UVICORN_ARGS=(serving.workflow_api:app --host 0.0.0.0 --port "$API_PORT")
if [[ "${DOCUFLOW_PROD:-0}" != "1" ]]; then
  UVICORN_ARGS+=(--reload)
else
  info "Production mode (DOCUFLOW_PROD=1) — uvicorn without --reload"
fi

uvicorn "${UVICORN_ARGS[@]}" &
UVICORN_PID=$!
info "API PID: $UVICORN_PID"

# ── vLLM watchdog (prod): exit non-zero so systemd Restart=on-failure restarts stack
WATCH_INTERVAL="${DOCUFLOW_VLLM_WATCHDOG_INTERVAL:-30}"
UNHEALTHY_LIMIT="${DOCUFLOW_VLLM_WATCHDOG_UNHEALTHY_LIMIT:-3}"
info "vLLM watchdog: every ${WATCH_INTERVAL}s, restart after ${UNHEALTHY_LIMIT} failed health checks"
unhealthy=0
while kill -0 "$UVICORN_PID" 2>/dev/null; do
    sleep "$WATCH_INTERVAL"
    [[ "$SHUTTING_DOWN" == "true" ]] && break

    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
        err "vLLM OCR process exited — stopping API for systemd restart"
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
        exit 1
    fi

    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        unhealthy=0
        continue
    fi

    unhealthy=$((unhealthy + 1))
    err "vLLM /health failed (${unhealthy}/${UNHEALTHY_LIMIT}) — log: $VLLM_LOG"
    if [[ "$unhealthy" -ge "$UNHEALTHY_LIMIT" ]]; then
        err "vLLM unhealthy — stopping API for systemd restart"
        if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
            kill -TERM "$VLLM_PID" 2>/dev/null || true
        fi
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
        exit 1
    fi
done

wait "$UVICORN_PID"
exit $?
