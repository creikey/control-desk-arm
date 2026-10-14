#!/usr/bin/env bash
# Install the signaling server, coturn (STUN/TURN) and Caddy (HTTPS) on a Debian/Ubuntu VPS.
# Run as root from this directory, with the robot's token in /etc/arm-signal/robot-token.
# Usage: ./setup.sh [domain]   (defaults to <public-ip>.sslip.io, which needs no DNS setup)
set -euo pipefail
cd "$(dirname "$0")"

PUBLIC_IP=$(curl -4 -fsS https://api.ipify.org)
PRIVATE_IP=$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
DOMAIN=${1:-${PUBLIC_IP//./-}.sslip.io}
CONF=/etc/arm-signal
APP=/opt/arm-signal

test -s "$CONF/robot-token" || { echo "Put the robot's token in $CONF/robot-token first." >&2; exit 1; }
test -s "$CONF/turn-secret" || openssl rand -hex 32 > "$CONF/turn-secret"
chmod 600 "$CONF"/*

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -yq coturn caddy curl
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

id arm-signal > /dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin arm-signal
install -d -o arm-signal "$APP"
install -o arm-signal -m 644 signaling.py pyproject.toml uv.lock "$APP"
runuser -u arm-signal -- env UV_CACHE_DIR="$APP/.cache" uv sync --frozen --project "$APP" -q

cat > "$CONF/env" <<EOF
ROBOT_TOKEN=$(cat "$CONF/robot-token")
TURN_SECRET=$(cat "$CONF/turn-secret")
PUBLIC_IP=$PUBLIC_IP
EOF
chmod 600 "$CONF/env"

cat > /etc/systemd/system/arm-signal.service <<EOF
[Unit]
Description=Desk arm WebRTC signaling
After=network-online.target

[Service]
User=arm-signal
EnvironmentFile=$CONF/env
ExecStart=$APP/.venv/bin/python $APP/signaling.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

EXTERNAL_IP=$PUBLIC_IP
[ "$PRIVATE_IP" != "$PUBLIC_IP" ] && EXTERNAL_IP="$PUBLIC_IP/$PRIVATE_IP"
cat > /etc/turnserver.conf <<EOF
listening-port=3478
external-ip=$EXTERNAL_IP
realm=$DOMAIN
use-auth-secret
static-auth-secret=$(cat "$CONF/turn-secret")
fingerprint
no-cli
no-tls
no-dtls
no-multicast-peers
min-port=49152
max-port=49352
user-quota=8
total-quota=64
max-bps=2000000
denied-peer-ip=0.0.0.0-0.255.255.255
denied-peer-ip=10.0.0.0-10.255.255.255
denied-peer-ip=100.64.0.0-100.127.255.255
denied-peer-ip=127.0.0.0-127.255.255.255
denied-peer-ip=169.254.0.0-169.254.255.255
denied-peer-ip=172.16.0.0-172.31.255.255
denied-peer-ip=192.168.0.0-192.168.255.255
syslog
EOF
echo "TURNSERVER_ENABLED=1" > /etc/default/coturn

cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
	reverse_proxy 127.0.0.1:8765
}
EOF

if command -v ufw > /dev/null && ufw status | grep -q "Status: active"; then
    ufw allow 80,443/tcp
    ufw allow 3478
    ufw allow 49152:49352/udp
fi

systemctl daemon-reload
systemctl enable -q arm-signal coturn caddy
systemctl restart arm-signal coturn caddy

echo "Signaling server: wss://$DOMAIN"
