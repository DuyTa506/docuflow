#!/usr/bin/env bash
# Install DocuFlow auto-start on Linux boot (systemd + PM2).
#
# Stack (systemd):
#   docuflow-infra           → postgres, minio, temporal (docker compose)
#   docuflow-backend         → llama + vLLM OCR + uvicorn API (start.sh)
#   docuflow-temporal-worker → digest/translation/stage worker (Restart=always)
#   docuflow-extraction-worker → extraction worker; process duy nhất nạp Docling lên GPU
# Frontend: pm2 serve Fe-Library/dist → :4200
#
# Alternative: API+worker in Docker → bash deploy/install-docker-autostart.sh
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

install_systemd_unit() {
  local name="$1"
  local src="$ROOT/deploy/${name}.service"
  local dst="/etc/systemd/system/${name}.service"
  local tmp
  tmp="$(mktemp)"
  sed -e "s|@@DOCUFLOW_ROOT@@|$ROOT|g" \
      -e "s|@@SERVICE_USER@@|$SERVICE_USER|g" \
      -e "s|@@SERVICE_HOME@@|$SERVICE_HOME|g" \
      -e "s|@@CONDA_BASE@@|${CONDA_BASE:-}|g" \
      "$src" > "$tmp"
  info "Installing systemd unit → $dst"
  sudo cp "$tmp" "$dst"
  rm -f "$tmp"
  sudo systemctl enable "$name.service"
  ok "systemd: $name enabled"
}

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
[[ -x "$ROOT/scripts/start_temporal_worker.sh" ]] || die "Missing scripts/start_temporal_worker.sh"

NODE_BIN_DIR="$(find_node_bin)" || die "node not found — install Node (nvm: nvm install 20) or set NODE_BIN=/path/to/node"
NODE_BIN="$NODE_BIN_DIR/node"
info "Using node: $NODE_BIN"

PM2_JS="$ROOT/Fe-Library/node_modules/pm2/bin/pm2"
if [[ ! -f "$PM2_JS" ]]; then
  info "Installing pm2 in Fe-Library…"
  (cd "$ROOT/Fe-Library" && env PATH="$NODE_BIN_DIR:$PATH" npm install pm2 --no-save)
fi
[[ -f "$PM2_JS" ]] || die "pm2 not found under Fe-Library/node_modules"

command -v docker >/dev/null || die "docker not installed (required for infra + llama.cpp)"

# shellcheck disable=SC1091
source "$ROOT/scripts/conda_env.sh"
CONDA_BASE="$(find_conda_base)" || die "conda not found — expected ~/anaconda3 or ~/miniconda3 (or set CONDA_BASE)"
info "Using conda: $CONDA_BASE (env: vllm-blackwell)"

warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }

echo ""
echo -e "\033[0;36mDocuFlow stack (systemd, host API) will start on boot:\033[0m"
echo "  1. docuflow-infra           — postgres, minio, temporal (docker)"
echo "  2. docuflow-backend         — llama.cpp + vLLM OCR + API :8022"
echo "  3. docuflow-temporal-worker — digest/translation/stage (Temporal)"
echo "  4. docuflow-extraction-worker — extraction: Docling + OCR (giữ GPU)"
echo "  First boot after power-on may take several minutes while GPU models load."
echo ""

info "Pre-flight: docker + llama.cpp…"
if docker ps --format '{{.Names}}' | grep -q '^llamacpp-qwen3.5-9b$'; then
  ok "llamacpp-qwen3.5-9b already running"
elif docker ps -a --format '{{.Names}}' | grep -q '^llamacpp-qwen3.5-9b$'; then
  warn "llamacpp-qwen3.5-9b exists but stopped — start.sh will start it on backend boot"
else
  warn "llamacpp-qwen3.5-9b not created yet — start.sh will run docker compose on first backend start"
fi
echo ""

install_systemd_unit docuflow-infra
install_systemd_unit docuflow-backend
install_systemd_unit docuflow-temporal-worker
# Tách riêng: đây là process duy nhất nạp Docling lên GPU.
install_systemd_unit docuflow-extraction-worker

sudo systemctl daemon-reload

info "Enable docker on boot (if not already)…"
sudo systemctl enable docker 2>/dev/null || true

# ── PM2 frontend ──────────────────────────────────────────────────
info "Registering PM2 app docuflow-fe (port 4200)…"
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

# ── Start now (optional) ──────────────────────────────────────────
if [[ -t 0 ]]; then
  read -r -p "Start stack now (infra → backend → worker)? [y/N] " START_NOW
else
  START_NOW="${DOCUFLOW_START_NOW:-N}"
fi
if [[ "${START_NOW,,}" == "y" ]]; then
  sudo systemctl start docuflow-infra.service
  sudo systemctl start docuflow-backend.service
  sudo systemctl start docuflow-temporal-worker.service
  sudo systemctl start docuflow-extraction-worker.service
  ok "Stack starting…"
  warn "GPU model load takes time — run: bash deploy/check-backend.sh"
fi

echo ""
ok "Done."
echo "  FE         → http://localhost:4200  (pm2 list)"
echo "  API        → http://localhost:8022  (systemctl status docuflow-backend)"
echo "  Temporal UI→ http://localhost:8088"
echo ""
echo "Health check:"
echo "  bash deploy/check-backend.sh"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status docuflow-infra docuflow-backend docuflow-temporal-worker"
echo "  journalctl -u docuflow-temporal-worker -f"
echo "  bash deploy/stop-autostart.sh    # pause (keep units, disable auto)"
echo "  bash deploy/start-autostart.sh   # resume + re-enable auto"
echo "  bash deploy/uninstall-autostart.sh"
echo ""
echo "Docker packaging (API+worker in containers):"
echo "  bash deploy/install-docker-autostart.sh"
