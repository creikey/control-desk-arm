#!/usr/bin/env bash
# Deploy the signaling server to a VPS over SSH, using this PC's robot token.
# Usage: ./deploy.sh root@host [domain]
set -euo pipefail
cd "$(dirname "$0")"

HOST=$1
TOKEN_FILE="${APPDATA:-$HOME/.config}/control-desk-arm/robot-token"

ssh "$HOST" 'install -d -m 700 /etc/arm-signal /tmp/arm-signal && cat > /etc/arm-signal/robot-token' < "$TOKEN_FILE"
scp -q signaling.py pyproject.toml uv.lock setup.sh "$HOST:/tmp/arm-signal/"
ssh "$HOST" "bash /tmp/arm-signal/setup.sh ${2:-}"
