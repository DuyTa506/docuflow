#!/usr/bin/env bash
# Re-install only the systemd unit (e.g. after conda PATH fix) without touching PM2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="${USER}"
SERVICE_HOME="${HOME}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) SERVICE_USER="$2"; SERVICE_HOME="$(eval echo "~$2")"; shift 2 ;;
    *) echo "Usage: bash deploy/update-backend-service.sh [--user USER]"; exit 1 ;;
  esac
done

# shellcheck disable=SC1091
source "$ROOT/scripts/conda_env.sh"
CONDA_BASE="$(CONDA_BASE= HOME="$SERVICE_HOME" find_conda_base)" \
  || { echo "conda not found under $SERVICE_HOME"; exit 1; }

TMP="$(mktemp)"
sed -e "s|@@DOCUFLOW_ROOT@@|$ROOT|g" \
    -e "s|@@SERVICE_USER@@|$SERVICE_USER|g" \
    -e "s|@@SERVICE_HOME@@|$SERVICE_HOME|g" \
    -e "s|@@CONDA_BASE@@|$CONDA_BASE|g" \
    "$ROOT/deploy/docuflow-backend.service" > "$TMP"

echo "Installing docuflow-backend.service (CONDA_BASE=$CONDA_BASE)"
sudo cp "$TMP" /etc/systemd/system/docuflow-backend.service
rm -f "$TMP"
sudo systemctl daemon-reload
echo "Done. Start with: sudo systemctl restart docuflow-backend"
