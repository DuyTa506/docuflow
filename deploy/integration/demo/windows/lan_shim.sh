#!/usr/bin/env bash
# A host can't talk to its own macvlan children through the parent NIC. Add a
# macvlan "shim" on the host and route the VM addresses through it, keeping
# the host's LAN address as source so the VM sees traffic from 192.168.212.60.
# Not persistent: gone after a host reboot (or `sudo bash lan_shim.sh down`).
set -euo pipefail
PARENT=${PARENT:-enp0s31f6}
SRC=${SRC:-192.168.212.60}
ADDRS=(192.168.212.249 192.168.212.250)
case "${1:-up}" in
  up)
    ip link show lan212-shim >/dev/null 2>&1 || ip link add lan212-shim link "$PARENT" type macvlan mode bridge
    ip link set lan212-shim up
    for a in "${ADDRS[@]}"; do ip route replace "$a/32" dev lan212-shim src "$SRC"; done ;;
  down)
    for a in "${ADDRS[@]}"; do ip route del "$a/32" 2>/dev/null || true; done
    ip link del lan212-shim 2>/dev/null || true ;;
esac
