#!/usr/bin/env bash
# Wait until host:port accepts TCP connections (used by systemd / worker startup).
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:?port required}"
TIMEOUT="${3:-120}"

deadline=$((SECONDS + TIMEOUT))
while (( SECONDS < deadline )); do
  if (echo >/dev/tcp/"$HOST"/"$PORT") 2>/dev/null; then
    exit 0
  fi
  sleep 2
done
echo "Timeout waiting for ${HOST}:${PORT} (${TIMEOUT}s)" >&2
exit 1
