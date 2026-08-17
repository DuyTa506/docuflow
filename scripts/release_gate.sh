#!/usr/bin/env bash
# Release-gate evidence for LAN-first production.
#
# Automated: unit tests that encode admission, lease, Range, secrets, health.
# Manual (this host, with GPU): chaos / load / restore — print the checklist.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== unit tests =="
python -m pytest --override-ini="addopts=" -q \
  tests/unit/test_pipeline/test_admission.py \
  tests/unit/test_pipeline/test_gpu_lease.py \
  tests/unit/test_pipeline/test_capacity.py \
  tests/unit/test_utils/test_file_download_range.py \
  tests/unit/test_settings/test_production_secrets.py \
  tests/unit/test_services/test_registration_approval.py \
  tests/unit/test_main_content/test_chapter_checkpoint.py \
  tests/unit/test_health/test_health_routes.py

echo
echo "== manual GPU / chaos / DR (must be ticked before Go) =="
cat <<'EOF'
[ ] GPU load: 2 digest + 1 translation + 1 extraction together; no OOM/watchdog churn
[ ] Worker restart mid OCR page, translation unit, summarize node, main-content chapter — resume, no duplicate persist
[ ] 200 MB upload/download with Range; oversized upload 413; memory stays bounded
[ ] Restore last off-host backup into staging; original downloads; one OCR or digest runs
[ ] LAN scan from a user laptop: only :8022 open; default admin/admin rejected when DOCUFLOW_PROD=1
[ ] E2E: login → upload → OCR → translate → digest → download; MEMBER cannot see another user's document
EOF
