#!/usr/bin/env bash
# Install PolyAgent on a fresh Debian/Ubuntu VPS. Run as root from the repo checkout.
set -euo pipefail
APP=/opt/polyagent
id -u polyagent >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin polyagent
apt-get update -y && apt-get install -y python3 python3-venv curl
mkdir -p "$APP"
cp -r polyagent requirements.txt "$APP"/
[ -f "$APP/.env" ] || cp .env.example "$APP/.env"
mkdir -p "$APP/data"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"
chown -R polyagent:polyagent "$APP"
chmod 600 "$APP/.env"
cp deploy/*.service deploy/*.timer /etc/systemd/system/
systemctl daemon-reload
# The directional agent (polyagent.service) showed no out-of-sample edge, so the market
# maker runs by default. Never run both against the same database.
systemctl enable --now polyagent-mm polyagent-dashboard polyagent-watchdog.timer polyagent-report.timer
echo "Installed. Edit $APP/.env, then: systemctl restart polyagent-mm"
echo "Dashboard listens on 127.0.0.1:8080 - reach it with: ssh -L 8080:127.0.0.1:8080 <vps>"
