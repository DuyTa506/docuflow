#!/usr/bin/env bash
# Off-host-capable backup: PostgreSQL dump + MinIO prefix copy + checksums.
#
# Usage:
#   bash scripts/backup.sh [DEST_DIR]
#
# Default DEST_DIR is ./backups/YYYYMMDD_HHMMSS. Copy that directory off the
# GPU host (NAS, second disk, another LAN machine) — a backup that lives only
# on the same disk as postgres_data is not a backup.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="${1:-$ROOT/backups/$STAMP}"
mkdir -p "$DEST"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

echo "Backing up to $DEST"

if docker ps --format '{{.Names}}' | grep -qx docuflow-postgres; then
  docker exec docuflow-postgres pg_dump -U "${POSTGRES_USER:-docuflow}" \
    "${POSTGRES_DB:-docuflow}" > "$DEST/postgres.sql"
else
  echo "docuflow-postgres is not running — skipping pg_dump" >&2
  exit 1
fi

python "$ROOT/scripts/backup_minio.py" "$DEST/minio"

(
  cd "$DEST"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "Backup complete: $DEST"
echo "Copy this directory off-host, then: bash scripts/restore.sh $DEST"
