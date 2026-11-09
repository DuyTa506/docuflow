#!/usr/bin/env bash
# Install DocuFlow auto-start on Linux boot (systemd).
#
# Stack (systemd):
#   docuflow-infra           → postgres, minio, temporal (docker compose)
#   docuflow-backend         → llama + vLLM OCR + uvicorn API (start.sh)
#   docuflow-temporal-worker → digest/translation/stage worker (Restart=always)
#   docuflow-extraction-worker → extraction worker; process duy nhất nạp Docling lên GPU
# Frontend: served by the API itself from Fe-Library/dist (one port, :8022).
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

[[ -f "$ROOT/start.sh" ]] || die "Run from DocuFlow repo (missing start.sh)"
[[ -d "$ROOT/Fe-Library/dist" ]] || die "Missing Fe-Library/dist — build FE first (ng build)"
[[ -x "$ROOT/.venv/bin/uvicorn" ]] || die "Missing .venv — run: uv venv && uv pip install -r requirements.txt"
[[ -x "$ROOT/scripts/start_temporal_worker.sh" ]] || die "Missing scripts/start_temporal_worker.sh"

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
install_systemd_unit docuflow-backup
if [[ -f "$ROOT/deploy/docuflow-backup.timer" ]]; then
  tmp="$(mktemp)"
  sed -e "s|@@DOCUFLOW_ROOT@@|$ROOT|g" \
      -e "s|@@SERVICE_USER@@|$SERVICE_USER|g" \
      "$ROOT/deploy/docuflow-backup.timer" > "$tmp"
  sudo cp "$tmp" /etc/systemd/system/docuflow-backup.timer
  rm -f "$tmp"
  sudo systemctl enable docuflow-backup.timer
  ok "systemd: docuflow-backup.timer enabled (02:30 daily)"
fi

sudo systemctl daemon-reload

info "Enable docker on boot (if not already)…"
sudo systemctl enable docker 2>/dev/null || true

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
echo "  API+UI     → http://<host>:8022  (only LAN entrypoint)"
echo "  Temporal UI→ http://127.0.0.1:8088  (host loopback only)"
echo ""
echo "Health check:"
echo "  bash deploy/check-backend.sh"
echo "  curl -sf http://127.0.0.1:8022/health/ready"
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
