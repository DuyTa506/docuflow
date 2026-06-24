#!/usr/bin/env bash
# Remove DocuFlow auto-start (systemd backend + PM2 frontend).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PM2_JS="$ROOT/Fe-Library/node_modules/pm2/bin/pm2"
NODE_BIN="${NODE_BIN:-$(ls "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | tail -1)}"

echo "Stopping and disabling docuflow-backend…"
sudo systemctl stop docuflow-backend.service 2>/dev/null || true
sudo systemctl disable docuflow-backend.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/docuflow-backend.service
sudo systemctl daemon-reload

if [[ -f "$PM2_JS" && -n "$NODE_BIN" && -x "$NODE_BIN" ]]; then
  "$NODE_BIN" "$PM2_JS" delete docuflow-fe 2>/dev/null || true
  "$NODE_BIN" "$PM2_JS" save 2>/dev/null || true
fi

echo "Done. PM2 startup hook may still exist — run: pm2 unstartup systemd"
