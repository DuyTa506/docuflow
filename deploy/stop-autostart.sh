#!/usr/bin/env bash
# Temporarily stop DocuFlow (keep unit files). Does NOT uninstall.
# Disables auto-start so reboot won't bring the stack back while you use another repo.
# Resume later: bash deploy/start-autostart.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PM2_JS="$ROOT/Fe-Library/node_modules/pm2/bin/pm2"
NODE_BIN="${NODE_BIN:-$(ls "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | tail -1)}"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }

# Need interactive sudo (password). Fail loudly — do not swallow auth errors.
if ! sudo -v; then
  echo "sudo required to stop systemd units. Re-run in a terminal:"
  echo "  bash deploy/stop-autostart.sh"
  exit 1
fi

# Reverse dependency order: workers → backend → infra
for unit in docuflow-extraction-worker docuflow-temporal-worker docuflow-backend docuflow-infra; do
  if [[ -f "/etc/systemd/system/${unit}.service" ]]; then
    info "Stopping and disabling $unit…"
    sudo systemctl stop "${unit}.service"
    sudo systemctl disable "${unit}.service"
  else
    info "Skip $unit (unit not installed)"
  fi
done

# Docker-packaged tier (if present)
if [[ -f /etc/systemd/system/docuflow-docker-stack.service ]]; then
  info "Stopping and disabling docuflow-docker-stack…"
  sudo systemctl stop docuflow-docker-stack.service
  sudo systemctl disable docuflow-docker-stack.service
fi

if [[ -f /etc/systemd/system/docuflow-backup.timer ]]; then
  info "Stopping and disabling docuflow-backup.timer…"
  sudo systemctl stop docuflow-backup.timer || true
  sudo systemctl disable docuflow-backup.timer || true
fi

# Remove infra containers so Docker restart=unless-stopped cannot bring them
# back on daemon restart while the systemd unit is disabled.
info "docker compose down (postgres/minio/temporal)…"
docker compose -f "$ROOT/docker-compose.yml" down || true

info "Stopping llama.cpp compose (port 5011)…"
docker compose -f "$ROOT/SETUPS/llms/docker-compose.yml" down || true

if [[ -f "$PM2_JS" && -n "${NODE_BIN:-}" && -x "$NODE_BIN" ]]; then
  info "Removing PM2 frontend (docuflow-fe) from dump so it will not resurrect…"
  "$NODE_BIN" "$PM2_JS" delete docuflow-fe || true
  "$NODE_BIN" "$PM2_JS" save --force || true
else
  info "PM2 not found — stop FE manually: pm2 delete docuflow-fe && pm2 save"
fi

ok "DocuFlow stopped (units kept, auto-start disabled)."
echo "  Resume: bash deploy/start-autostart.sh"
echo "  Remove units entirely: bash deploy/uninstall-autostart.sh"
