#!/usr/bin/env bash
# Remove DocuFlow auto-start (systemd + PM2 frontend).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PM2_JS="$ROOT/Fe-Library/node_modules/pm2/bin/pm2"
NODE_BIN="${NODE_BIN:-$(ls "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | tail -1)}"

for unit in docuflow-extraction-worker docuflow-temporal-worker docuflow-backend docuflow-infra docuflow-docker-stack; do
  echo "Stopping and disabling $unit…"
  sudo systemctl stop "$unit.service" 2>/dev/null || true
  sudo systemctl disable "$unit.service" 2>/dev/null || true
  sudo rm -f "/etc/systemd/system/${unit}.service"
done
sudo systemctl daemon-reload

if [[ -f "$PM2_JS" && -n "$NODE_BIN" && -x "$NODE_BIN" ]]; then
  "$NODE_BIN" "$PM2_JS" delete docuflow-fe 2>/dev/null || true
  "$NODE_BIN" "$PM2_JS" save 2>/dev/null || true
fi

echo "Done. PM2 startup hook may still exist — run: pm2 unstartup systemd"
