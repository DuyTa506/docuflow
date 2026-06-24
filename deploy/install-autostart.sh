#!/usr/bin/env bash
# Install DocuFlow auto-start on Linux boot (systemd + PM2).
#
# Backend : start.sh via systemd  → API :8002, vLLM :8000, llama docker
# Frontend: pm2 serve Fe-Library/dist → :4200
#
# Usage:
#   bash deploy/install-autostart.sh
#   bash deploy/install-autostart.sh --user dell
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="${USER}"
SERVICE_HOME="${HOME}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) SERVICE_USER="$2"; SERVICE_HOME="$(eval echo "~$2")"; shift 2 ;;
    -h|--help)
      echo "Usage: bash deploy/install-autostart.sh [--user USERNAME]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
die()   { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

# PM2 is a Node script; sudo -u does not load nvm/bashrc — resolve node explicitly.
find_node_bin() {
  if [[ -n "${NODE_BIN:-}" && -x "$NODE_BIN" ]]; then
    echo "$(dirname "$NODE_BIN")"
    return 0
  fi
  local candidate node_path
  for candidate in \
    "$SERVICE_HOME/.nvm/versions/node/"*/bin/node \
    /usr/local/bin/node \
    /usr/bin/node; do
    [[ -x "$candidate" ]] || continue
    echo "$(dirname "$candidate")"
    return 0
  done
  if [[ -s "$SERVICE_HOME/.nvm/nvm.sh" ]]; then
    node_path="$(sudo -u "$SERVICE_USER" bash -c "source '$SERVICE_HOME/.nvm/nvm.sh' && command -v node" 2>/dev/null || true)"
    if [[ -n "$node_path" && -x "$node_path" ]]; then
      echo "$(dirname "$node_path")"
      return 0
    fi
  fi
  return 1
}

run_pm2() {
  sudo -u "$SERVICE_USER" env \
    HOME="$SERVICE_HOME" \
    PATH="$NODE_BIN_DIR:$PATH" \
    "$NODE_BIN" "$PM2_JS" "$@"
}

[[ -f "$ROOT/start.sh" ]] || die "Run from DocuFlow repo (missing start.sh)"
[[ -d "$ROOT/Fe-Library/dist" ]] || die "Missing Fe-Library/dist — build FE first (ng build)"
[[ -x "$ROOT/.venv/bin/uvicorn" ]] || die "Missing .venv — run: uv venv && uv pip install -r requirements.txt"

NODE_BIN_DIR="$(find_node_bin)" || die "node not found — install Node (nvm: nvm install 20) or set NODE_BIN=/path/to/node"
NODE_BIN="$NODE_BIN_DIR/node"
info "Using node: $NODE_BIN"

PM2_JS="$ROOT/Fe-Library/node_modules/pm2/bin/pm2"
if [[ ! -f "$PM2_JS" ]]; then
  info "Installing pm2 in Fe-Library…"
  (cd "$ROOT/Fe-Library" && env PATH="$NODE_BIN_DIR:$PATH" npm install pm2 --no-save)
fi
[[ -f "$PM2_JS" ]] || die "pm2 not found under Fe-Library/node_modules"

command -v docker >/dev/null || die "docker not installed (required for llama.cpp container)"

# shellcheck disable=SC1091
source "$ROOT/scripts/conda_env.sh"
CONDA_BASE="$(find_conda_base)" || die "conda not found — expected ~/anaconda3 or ~/miniconda3 (or set CONDA_BASE)"
info "Using conda: $CONDA_BASE (env: vllm-blackwell)"

warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }

echo ""
echo -e "\033[0;36mBackend (docuflow-backend) will start on boot:\033[0m"
echo "  1. llama.cpp  — docker container llamacpp-qwen3.5-9b  (pipeline LLM, host :5011)"
echo "  2. DeepSeek-OCR-2 — vLLM via conda env vllm-blackwell (OCR, :8000)"
echo "     same command as: bash serve_deepseek_ocr.sh"
echo "  3. DocuFlow API — uvicorn :8002"
echo "  First boot after power-on may take several minutes while GPU models load."
echo ""

info "Pre-flight: docker + llama.cpp…"
if docker ps --format '{{.Names}}' | grep -q '^llamacpp-qwen3.5-9b$'; then
  ok "llamacpp-qwen3.5-9b already running"
elif docker ps -a --format '{{.Names}}' | grep -q '^llamacpp-qwen3.5-9b$'; then
  warn "llamacpp-qwen3.5-9b exists but stopped — start.sh will start it on backend boot"
else
  warn "llamacpp-qwen3.5-9b not created yet — start.sh will run docker compose on first backend start"
  warn "  Ensure model files exist under SETUPS/llms/models/"
fi
echo ""
UNIT_SRC="$ROOT/deploy/docuflow-backend.service"
UNIT_DST="/etc/systemd/system/docuflow-backend.service"
TMP_UNIT="$(mktemp)"
sed -e "s|@@DOCUFLOW_ROOT@@|$ROOT|g" \
    -e "s|@@SERVICE_USER@@|$SERVICE_USER|g" \
    -e "s|@@SERVICE_HOME@@|$SERVICE_HOME|g" \
    -e "s|@@CONDA_BASE@@|$CONDA_BASE|g" \
    "$UNIT_SRC" > "$TMP_UNIT"

info "Installing systemd unit → $UNIT_DST (sudo)"
sudo cp "$TMP_UNIT" "$UNIT_DST"
rm -f "$TMP_UNIT"
sudo systemctl daemon-reload
sudo systemctl enable docuflow-backend.service
ok "systemd: docuflow-backend enabled (starts on boot)"

info "Enable docker on boot (if not already)…"
sudo systemctl enable docker 2>/dev/null || true

# ── 2. PM2 frontend ──────────────────────────────────────────────────
info "Registering PM2 app docuflow-fe (port 4200)…"
# FE dev may have run `pm2 serve dist 4200` earlier — free the port first
run_pm2 delete static-page-server-4200 2>/dev/null || true
run_pm2 delete docuflow-fe 2>/dev/null || true
run_pm2 start "$ROOT/deploy/ecosystem.config.cjs"
run_pm2 save

info "PM2 startup hook (resurrect after reboot)…"
PM2_STARTUP_LOG="$(mktemp)"
run_pm2 startup systemd -u "$SERVICE_USER" --hp "$SERVICE_HOME" \
  >"$PM2_STARTUP_LOG" 2>&1 || true
if grep -qE '^sudo ' "$PM2_STARTUP_LOG"; then
  grep -E '^sudo ' "$PM2_STARTUP_LOG" | bash || info "Run the sudo command from: $PM2_STARTUP_LOG"
else
  cat "$PM2_STARTUP_LOG"
fi
rm -f "$PM2_STARTUP_LOG"
ok "PM2: docuflow-fe saved for auto-resurrect"

# ── 3. Start now (optional) ──────────────────────────────────────────
if [[ -t 0 ]]; then
  read -r -p "Start backend now? [y/N] " START_NOW
else
  START_NOW="${DOCUFLOW_START_NOW:-N}"
fi
if [[ "${START_NOW,,}" == "y" ]]; then
  sudo systemctl start docuflow-backend.service
  ok "Backend starting (docker llama + DeepSeek OCR vLLM + API)…"
  warn "GPU model load takes time — run: bash deploy/check-backend.sh"
  warn "Live log: journalctl -u docuflow-backend -f"
fi

echo ""
ok "Done."
echo "  FE  → http://localhost:4200  (pm2 list)"
echo "  BE  → http://localhost:8002  (after models load; systemctl status docuflow-backend)"
echo ""
echo "Backend stack (auto-started by docuflow-backend / start.sh):"
echo "  • llamacpp-qwen3.5-9b  docker  → :5011  (translate, summarize, …)"
echo "  • DeepSeek-OCR-2       vLLM     → :8000  (bash serve_deepseek_ocr.sh)"
echo "  • DocuFlow API         uvicorn  → :8002"
echo ""
echo "Health check:"
echo "  bash deploy/check-backend.sh"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status docuflow-backend"
echo "  journalctl -u docuflow-backend -f"
echo "  pm2 logs docuflow-fe"
echo "  bash deploy/uninstall-autostart.sh"
