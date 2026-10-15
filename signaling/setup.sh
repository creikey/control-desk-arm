#!/usr/bin/env bash
# Install the signaling server, coturn (STUN/TURN) and Caddy (HTTPS) on a Debian/Ubuntu VPS.
# HAProxy owns port 443 and splits it by TLS server name: turn.<domain> goes to coturn (TURN over TLS,
# for networks that only allow web traffic), everything else to Caddy.
# Run as root from this directory, with the robot's token in /etc/arm-signal/robot-token.
# Usage: ./setup.sh [domain]   (defaults to <public-ip>.sslip.io, which needs no DNS setup;
#                               a custom domain also needs turn.<domain> pointing here)
set -euo pipefail
cd "$(dirname "$0")"

PUBLIC_IP=$(curl -4 -fsS https://api.ipify.org)
PRIVATE_IP=$(ip -4 route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
DOMAIN=${1:-${PUBLIC_IP//./-}.sslip.io}
TURN_DOMAIN=turn.$DOMAIN
CONF=/etc/arm-signal
APP=/opt/arm-signal

test -s "$CONF/robot-token" || { echo "Put the robot's token in $CONF/robot-token first." >&2; exit 1; }
test -s "$CONF/turn-secret" || openssl rand -hex 32 > "$CONF/turn-secret"
chmod 600 "$CONF"/*

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -yq coturn caddy haproxy curl
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

id arm-signal > /dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin arm-signal
install -d -o arm-signal "$APP"
install -o arm-signal -m 644 signaling.py pyproject.toml uv.lock "$APP"
runuser -u arm-signal -- env UV_CACHE_DIR="$APP/.cache" uv sync --frozen --project "$APP" -q

cat > "$CONF/env" <<EOF
ROBOT_TOKEN=$(cat "$CONF/robot-token")
TURN_SECRET=$(cat "$CONF/turn-secret")
PUBLIC_IP=$PUBLIC_IP
TURN_DOMAIN=$TURN_DOMAIN
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
tls-listening-port=5349
cert=/etc/coturn/turn.crt
pkey=/etc/coturn/turn.key
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
{
	https_port 8443
}

$DOMAIN {
	reverse_proxy 127.0.0.1:8765
}

# Only here so Caddy keeps a certificate for coturn.
$TURN_DOMAIN {
	respond 404
}
EOF

cat > /etc/haproxy/haproxy.cfg <<EOF
global
	log /dev/log local0

defaults
	mode tcp
	log global
	timeout connect 5s
	timeout client 1h
	timeout server 1h

frontend tls
	bind :443
	tcp-request inspect-delay 5s
	tcp-request content accept if { req_ssl_hello_type 1 }
	use_backend coturn if { req_ssl_sni -i $TURN_DOMAIN }
	default_backend caddy

backend caddy
	server caddy 127.0.0.1:8443

backend coturn
	server coturn 127.0.0.1:5349
EOF

# coturn can't read Caddy's certificates directly, so copy them over whenever Caddy renews.
cat > /usr/local/sbin/arm-turn-cert <<EOF
#!/bin/sh
set -eu
crt=\$(find /var/lib/caddy -name '$TURN_DOMAIN.crt' -printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-)
test -n "\$crt" || { echo "no certificate for $TURN_DOMAIN yet" >&2; exit 1; }
cmp -s "\$crt" /etc/coturn/turn.crt && exit 0
install -D -m 644 "\$crt" /etc/coturn/turn.crt
install -D -m 640 -g turnserver "\${crt%.crt}.key" /etc/coturn/turn.key
systemctl restart coturn
EOF
chmod 755 /usr/local/sbin/arm-turn-cert

cat > /etc/systemd/system/arm-turn-cert.service <<EOF
[Unit]
Description=Copy the TURN certificate from Caddy to coturn

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/arm-turn-cert
EOF

cat > /etc/systemd/system/arm-turn-cert.timer <<EOF
[Unit]
Description=Copy the TURN certificate from Caddy to coturn daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

if command -v ufw > /dev/null && ufw status | grep -q "Status: active"; then
    ufw allow 80,443/tcp
    ufw allow 3478
    ufw allow 49152:49352/udp
fi

systemctl daemon-reload
systemctl enable -q arm-signal coturn caddy haproxy arm-turn-cert.timer
systemctl restart arm-signal coturn caddy  # Caddy lets go of 443 before HAProxy takes it
systemctl restart haproxy
systemctl start arm-turn-cert.timer

echo "Waiting for Caddy's certificate for $TURN_DOMAIN…"
for _ in $(seq 60); do /usr/local/sbin/arm-turn-cert 2> /dev/null && break; sleep 2; done
/usr/local/sbin/arm-turn-cert

echo "Signaling server: wss://$DOMAIN"
