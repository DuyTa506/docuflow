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
    # Kill vLLM OCR if we started it — the whole group, so EngineCore does not
    # survive us holding 9 GB of VRAM. (The early `exit`s above run this trap
    # before stop_vllm_ocr is defined; VLLM_PID is empty then anyway.)
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        info "Stopping vLLM OCR server (pid $VLLM_PID)…"
        if declare -F stop_vllm_ocr >/dev/null; then
            stop_vllm_ocr
        else
            kill "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
        fi
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

# ── 3. vLLM OCR server (background, conda env) ───────────────────────
# Spawning lives in a function because the watchdog respawns it too. Only
# OCR/extraction needs this server: digest and translation talk to llama.cpp
# on :5011 and never touch :8000.
spawn_vllm_ocr() {
    # `set -m` puts the job in its own process group (PGID == $!), which is what
    # makes stop_vllm_ocr able to kill vLLM's children too. vLLM runs its engine
    # in a separate EngineCore process: signalling only the parent orphans that
    # child onto init, where it keeps its whole GPU allocation and every respawn
    # then dies with "Free memory on device cuda:0 (2.81/31.36 GiB) ... less than
    # desired (0.3, 9.41 GiB)".
    #
    # The subshell is because vLLM needs the conda env while the API needs
    # .venv; activating conda here would leave the parent on the wrong
    # interpreter after every respawn. `exec` keeps $! pointing at vllm itself.
    set -m
    (
        # shellcheck disable=SC1091
        source "$ROOT/scripts/conda_env.sh"
        activate_docuflow_conda || exit 1
        # shellcheck disable=SC1091
        source "$ROOT/scripts/vllm_ocr_config.sh"
        vllm_ocr_build_serve_args
        exec vllm serve "${VLLM_OCR_SERVE_ARGS[@]}"
    ) >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    set +m
    info "vLLM OCR PID: $VLLM_PID (log: $VLLM_LOG)"
}

# Stop vLLM and everything it spawned, then wait for the GPU to come back.
stop_vllm_ocr() {
    [[ -z "$VLLM_PID" ]] && return 0
    local pgid="$VLLM_PID"

    # Negative PID = the whole process group, so EngineCore goes too. The group
    # outlives its dead leader, which is the case that matters here: by the time
    # the watchdog notices, the parent is usually already gone and EngineCore is
    # the thing still holding the VRAM.
    kill -TERM -"$pgid" 2>/dev/null || kill -TERM "$pgid" 2>/dev/null || true

    # Freeing VRAM is not instant, and respawning into a half-released GPU fails
    # exactly like a GPU that is genuinely full. Wait on the whole group, not on
    # $VLLM_PID: the process holding the memory is EngineCore, a different pid.
    local gone=false
    for _ in $(seq 1 30); do
        if ! pgrep -g "$pgid" >/dev/null 2>&1; then
            gone=true
            break
        fi
        sleep 1
    done
    if [[ "$gone" != "true" ]]; then
        kill -KILL -"$pgid" 2>/dev/null || true
        sleep 3
    fi
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
}

# Wait up to $1 seconds for :8000 to answer /health.
wait_for_vllm_ocr() {
    local limit="$1"
    for _ in $(seq 1 "$limit"); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

info "Checking vLLM OCR server on :8000…"

if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    ok "vLLM OCR already healthy on :8000 — skipping spawn"
    VLLM_PID=""
else
    info "Starting vLLM OCR server (background)…"

    # Optional: structure-preserving export (PDF overlay + DocLayout ONNX)
    # shellcheck disable=SC1091
    source "$ROOT/scripts/conda_env.sh"
    activate_docuflow_conda || exit 1
    if ! python -c "import onnxruntime" 2>/dev/null; then
        info "onnxruntime not installed — PDF overlay translation disabled until: pip install onnxruntime onnx opencv-python-headless"
    fi
    if ! command -v pandoc >/dev/null 2>&1; then
        info "pandoc not found — LaTeX→OMML in DOCX export will use python-docx fallback"
    fi

    spawn_vllm_ocr || err "Could not spawn vLLM OCR — see $VLLM_LOG"

    info "Waiting for DeepSeek-OCR-2 vLLM on :8000 (same as serve_deepseek_ocr.sh)…"
    if wait_for_vllm_ocr 120; then
        ok "vLLM OCR ready"
    else
        err "DeepSeek-OCR-2 vLLM did not become ready within 120s."
        err "  Log: $VLLM_LOG"
        err "  Manual: bash serve_deepseek_ocr.sh"
        err "  API starts anyway — OCR/extract fails until vLLM is healthy, the rest works."
    fi
fi

# ── 4. API server (.venv, foreground) ────────────────────────────────
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
API_HOST="${API_HOST:-0.0.0.0}"
UVICORN_ARGS=(serving.workflow_api:app --host "$API_HOST" --port "$API_PORT")
if [[ "${DOCUFLOW_PROD:-0}" != "1" ]]; then
  UVICORN_ARGS+=(--reload)
else
  info "Production mode (DOCUFLOW_PROD=1) — uvicorn without --reload"
fi

uvicorn "${UVICORN_ARGS[@]}" &
UVICORN_PID=$!
info "API PID: $UVICORN_PID"

# ── vLLM watchdog: revive OCR in place, never take the API down ───────
#
# This used to `exit 1` so systemd would rebuild the whole stack. That works
# only while vLLM *can* come back. When it cannot — another project's vLLM
# took the GPU on 2026-08-11 at 15:00:55 and DeepSeek-OCR could no longer fit
# — every restart failed the same way: 317 of them, and each one spent four
# minutes on dependency checks before uvicorn even started. The API was down
# far more than it was up, and users clicking Tổng thuật or Dịch thuật (which
# never touch :8000) got network errors from a backend that had no reason to
# be dead.
#
# So: respawn only vLLM, keep serving everything else, and back off instead of
# hammering a GPU that has no room. OCR requests fail with a real error until
# it recovers; nothing else notices.
WATCH_INTERVAL="${DOCUFLOW_VLLM_WATCHDOG_INTERVAL:-30}"
UNHEALTHY_LIMIT="${DOCUFLOW_VLLM_WATCHDOG_UNHEALTHY_LIMIT:-3}"
RESPAWN_BACKOFF_MAX="${DOCUFLOW_VLLM_RESPAWN_BACKOFF_MAX:-600}"
info "vLLM watchdog: every ${WATCH_INTERVAL}s, respawn OCR after ${UNHEALTHY_LIMIT} failed health checks (API stays up)"

unhealthy=0
backoff=0
cooldown_until=0

revive_vllm_ocr() {
    local now
    now=$(date +%s)
    # Still cooling down from a failed respawn — say nothing, do nothing.
    if [[ "$now" -lt "$cooldown_until" ]]; then
        return 0
    fi

    # Always clear the group first, even when the parent already died: the
    # EngineCore child usually outlives it and holds the VRAM the respawn needs.
    stop_vllm_ocr

    info "Respawning vLLM OCR (API keeps serving on :${API_PORT})…"
    if spawn_vllm_ocr && wait_for_vllm_ocr 120; then
        ok "vLLM OCR recovered"
        unhealthy=0
        backoff=0
        cooldown_until=0
        return 0
    fi

    # Back off so a GPU with no room is not hammered every 30s. The API is
    # unaffected either way, so waiting costs nothing but OCR latency.
    if [[ "$backoff" -eq 0 ]]; then
        backoff="$WATCH_INTERVAL"
    else
        backoff=$(( backoff * 2 ))
    fi
    if [[ "$backoff" -gt "$RESPAWN_BACKOFF_MAX" ]]; then
        backoff="$RESPAWN_BACKOFF_MAX"
    fi
    cooldown_until=$(( $(date +%s) + backoff ))
    err "vLLM OCR still down — retrying in ${backoff}s. OCR/extract unavailable; digest & translation unaffected. Log: $VLLM_LOG"
    unhealthy=0
    return 0
}

while kill -0 "$UVICORN_PID" 2>/dev/null; do
    sleep "$WATCH_INTERVAL"
    [[ "$SHUTTING_DOWN" == "true" ]] && break

    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
        err "vLLM OCR process exited — reviving it, API untouched"
        revive_vllm_ocr
        continue
    fi

    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        unhealthy=0
        continue
    fi

    unhealthy=$((unhealthy + 1))
    err "vLLM /health failed (${unhealthy}/${UNHEALTHY_LIMIT}) — log: $VLLM_LOG"
    if [[ "$unhealthy" -ge "$UNHEALTHY_LIMIT" ]]; then
        revive_vllm_ocr
    fi
done

# Only the API dying ends the script — that is the one failure systemd should
# rebuild the stack for.
wait "$UVICORN_PID"
exit $?
