#!/usr/bin/env bash
# Re-enable and start DocuFlow after deploy/stop-autostart.sh.
# Requires units already installed (bash deploy/install-autostart.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PM2_JS="$ROOT/Fe-Library/node_modules/pm2/bin/pm2"
NODE_BIN="${NODE_BIN:-$(ls "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | tail -1)}"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
die()  { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

need_unit() {
  local unit="$1"
  [[ -f "/etc/systemd/system/${unit}.service" ]] || die "$unit.service missing — run: bash deploy/install-autostart.sh"
}

if ! sudo -v; then
  echo "sudo required to start systemd units. Re-run in a terminal:"
  echo "  bash deploy/start-autostart.sh"
  exit 1
fi

# Prefer host tier if those units exist; else docker stack.
if [[ -f /etc/systemd/system/docuflow-backend.service ]]; then
  for unit in docuflow-infra docuflow-backend docuflow-temporal-worker; do
    need_unit "$unit"
    info "Enabling and starting $unit…"
    sudo systemctl enable "${unit}.service"
    sudo systemctl start "${unit}.service"
  done
elif [[ -f /etc/systemd/system/docuflow-docker-stack.service ]]; then
  info "Enabling and starting docuflow-docker-stack…"
  sudo systemctl enable docuflow-docker-stack.service
  sudo systemctl start docuflow-docker-stack.service
else
  die "No DocuFlow units found — run: bash deploy/install-autostart.sh"
fi

if [[ -f "$PM2_JS" && -n "${NODE_BIN:-}" && -x "$NODE_BIN" ]]; then
  info "Starting PM2 frontend (docuflow-fe)…"
  "$NODE_BIN" "$PM2_JS" start docuflow-fe \
    || "$NODE_BIN" "$PM2_JS" resurrect \
    || true
  "$NODE_BIN" "$PM2_JS" save || true
else
  info "PM2 not found — start FE manually: pm2 start docuflow-fe"
fi

ok "DocuFlow starting (auto-start re-enabled)."
echo "  Health: bash deploy/check-backend.sh"
echo "  Status: sudo systemctl status docuflow-infra docuflow-backend docuflow-temporal-worker"
