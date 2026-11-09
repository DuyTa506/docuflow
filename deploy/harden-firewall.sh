#!/usr/bin/env bash
# Host firewall for the security-scan window: the API port is the only way in.
# Additive and idempotent: it never resets rules someone else added (a library
# workstation's own 8022 rule, say) — it sets the default policy and adds ours.
#
#   sudo bash deploy/harden-firewall.sh --no-ssh
#   sudo bash deploy/harden-firewall.sh --ssh-from 10.0.0.0/24
#   sudo bash deploy/harden-firewall.sh --ssh-from tailscale0     # an interface name
#
# SSH is never opened implicitly: say who may reach it, or pass --no-ssh. Docker
# publishes its ports around ufw, which is why every compose port is bound to
# 127.0.0.1 as well — the firewall is the second lock, not the only one.
set -euo pipefail

API_PORT="${API_PORT:-8022}"
SSH_FROM=""
NO_SSH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-from) SSH_FROM="$2"; shift 2 ;;
    --no-ssh) NO_SSH=1; shift ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }
if [[ -z "$SSH_FROM" && "$NO_SSH" -eq 0 ]]; then
  echo "Choose: --ssh-from <cidr|interface> or --no-ssh (refusing to guess, you could lock yourself out)." >&2
  exit 1
fi
command -v ufw >/dev/null || { echo "ufw not installed: apt install ufw" >&2; exit 1; }

ufw default deny incoming
ufw default allow outgoing
ufw allow "${API_PORT}/tcp" comment 'DocuFlow API + UI'
if [[ -n "$SSH_FROM" ]]; then
  if [[ "$SSH_FROM" == */* ]]; then
    ufw allow from "$SSH_FROM" to any port 22 proto tcp comment 'ssh (admin)'
  else
    ufw allow in on "$SSH_FROM" to any port 22 proto tcp comment 'ssh (admin)'
  fi
fi
ufw --force enable
ufw status verbose
echo
echo "Any other 'ALLOW IN' above is a rule this script did not add. Remove the ones you do not need:"
echo "  sudo ufw delete allow <port>/tcp"
