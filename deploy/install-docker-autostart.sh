#!/usr/bin/env bash
# Install DocuFlow Docker stack auto-start (systemd → docker compose --profile app).
#
# Runs infra + API + Temporal worker in Docker with restart: unless-stopped.
# GPU services (vLLM OCR :8000, llama.cpp :5011) must run on the HOST — see deploy/docker.env.example
#
# Usage:
#   bash deploy/install-docker-autostart.sh
#   bash deploy/install-docker-autostart.sh --user dell
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="${USER}"
SERVICE_HOME="${HOME}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) SERVICE_USER="$2"; SERVICE_HOME="$(eval echo "~$2")"; shift 2 ;;
    -h|--help)
      echo "Usage: bash deploy/install-docker-autostart.sh [--user USERNAME]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

[[ -f "$ROOT/docker-compose.yml" ]] || die "Run from DocuFlow repo"
command -v docker >/dev/null || die "docker not installed"

if [[ ! -f "$ROOT/.env" ]]; then
  warn "No .env — copy deploy/docker.env.example and merge with your secrets (JWT_SECRET_KEY, …)"
fi

echo ""
echo -e "${CYAN}Docker stack (docuflow-docker-stack) will start on boot:${NC}"
echo "  • postgres, minio, temporal, temporal-ui  (infra)"
echo "  • docuflow-api                            (:8022)"
echo "  • docuflow-temporal-worker                (digest pipeline)"
echo ""
warn "Host GPU stack still required for OCR/LLM:"
echo "  • vLLM OCR     → bash serve_deepseek_ocr.sh  (:8000)"
echo "  • llama.cpp    → = docker compose -f SETUPS/llms/docker-compose.yml up -d qwen3.5-9b  (:5011)"
echo ""
warn "Disable host API/worker if previously installed:"
echo "  sudo systemctl disable --now docuflow-backend docuflow-temporal-worker 2>/dev/null || true"
echo ""

UNIT_SRC="$ROOT/deploy/docuflow-docker-stack.service"
UNIT_DST="/etc/systemd/system/docuflow-docker-stack.service"
TMP_UNIT="$(mktemp)"
sed -e "s|@@DOCUFLOW_ROOT@@|$ROOT|g" \
    -e "s|@@SERVICE_USER@@|$SERVICE_USER|g" \
    -e "s|@@SERVICE_HOME@@|$SERVICE_HOME|g" \
    "$UNIT_SRC" > "$TMP_UNIT"

info "Installing systemd unit → $UNIT_DST (sudo)"
sudo cp "$TMP_UNIT" "$UNIT_DST"
rm -f "$TMP_UNIT"
sudo systemctl daemon-reload
sudo systemctl enable docuflow-docker-stack.service
ok "systemd: docuflow-docker-stack enabled"

info "Enable docker on boot…"
sudo systemctl enable docker 2>/dev/null || true

if [[ -t 0 ]]; then
  read -r -p "Start Docker stack now? [y/N] " START_NOW
else
  START_NOW="${DOCUFLOW_START_NOW:-N}"
fi
if [[ "${START_NOW,,}" == "y" ]]; then
  sudo systemctl start docuflow-docker-stack.service
  ok "Stack starting — first build may take several minutes"
  warn "Run: bash deploy/check-backend.sh --docker"
fi

echo ""
ok "Done."
echo "  API        → http://localhost:8022"
echo "  Temporal UI→ http://localhost:8088"
echo ""
echo "Commands:"
echo "  sudo systemctl status docuflow-docker-stack"
echo "  docker compose --profile app ps"
echo "  docker compose --profile app logs -f worker"
echo "  bash deploy/check-backend.sh --docker"
