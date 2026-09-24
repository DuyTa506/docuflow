#!/usr/bin/env bash
# Restore PostgreSQL + MinIO from a directory produced by scripts/backup.sh.
#
# Order: Postgres first (metadata), then MinIO objects. The API/workers must
# be stopped so nothing writes during restore.
#
# Usage:
#   bash scripts/restore.sh BACKUP_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "usage: bash scripts/restore.sh BACKUP_DIR" >&2
  exit 1
fi

if [[ -f "$SRC/SHA256SUMS" ]]; then
  (cd "$SRC" && sha256sum -c SHA256SUMS)
fi

if [[ ! -f "$SRC/postgres.sql" ]]; then
  echo "missing $SRC/postgres.sql" >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx docuflow-postgres; then
  echo "docuflow-postgres is not running" >&2
  exit 1
fi

echo "Restoring PostgreSQL…"
docker exec -i docuflow-postgres psql -U "${POSTGRES_USER:-docuflow}" \
  -d "${POSTGRES_DB:-docuflow}" < "$SRC/postgres.sql"

if [[ -d "$SRC/minio" ]]; then
  echo "Restoring MinIO objects…"
  python "$ROOT/scripts/restore_minio.py" "$SRC/minio"
fi

echo "Restore complete. Start API/workers and hit GET /health/ready."
echo "Then open one document, download the original, and run a short OCR/digest."
