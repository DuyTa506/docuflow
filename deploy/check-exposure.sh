#!/usr/bin/env bash
# Lists every TCP socket listening beyond loopback. For DocuFlow that must be the
# API port and nothing else (SSH aside, if you opened it on purpose).
#
#   bash deploy/check-exposure.sh            # exit 1 if anything else listens
set -euo pipefail

API_PORT="${API_PORT:-8022}"
extra=0
while read -r addr; do
  host="${addr%:*}"; port="${addr##*:}"
  case "$host" in 127.*|"[::1]"|::1|"127.0.0.53%lo"|"127.0.0.54"|"127.0.0.53") continue ;; esac
  [[ "$port" == "$API_PORT" ]] && { echo "OK    $addr  (DocuFlow API)"; continue; }
  echo "OPEN  $addr"; extra=1
done < <(ss -H -ltn | awk '{print $4}' | sort -u)

if [[ "$extra" -eq 1 ]]; then
  echo "-> listeners above are reachable from the network; close them or firewall them (deploy/harden-firewall.sh)."
  exit 1
fi
echo "Only :${API_PORT} listens beyond loopback."
