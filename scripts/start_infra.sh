#!/usr/bin/env bash
# Start DocuFlow infrastructure containers (Postgres, MinIO, Temporal).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi

docker compose -f "$ROOT/docker-compose.yml" up -d postgres minio temporal temporal-ui

bash "$ROOT/scripts/wait_for_port.sh" localhost 5433 60 || {
  echo "Postgres not ready on :5433" >&2
  exit 1
}
bash "$ROOT/scripts/wait_for_port.sh" localhost 9000 60 || {
  echo "MinIO not ready on :9000" >&2
  exit 1
}
bash "$ROOT/scripts/wait_for_port.sh" localhost 7233 120 || {
  echo "Temporal not ready on :7233" >&2
  exit 1
}

echo "Infrastructure ready (postgres, minio, temporal, temporal-ui)"
