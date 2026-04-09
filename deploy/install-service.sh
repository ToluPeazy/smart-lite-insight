#!/usr/bin/env bash
# Install Smart-Lite Insight as a systemd service on Raspberry Pi.
#
# Usage:
#   sudo bash deploy/install-service.sh [INSTALL_DIR]
#
# Defaults INSTALL_DIR to /opt/smart-lite-insight.
# Must be run as root (or with sudo).

set -euo pipefail

INSTALL_DIR="${1:-/opt/smart-lite-insight}"
SERVICE_NAME="smartlite"
SERVICE_FILE="deploy/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"

# ── Preflight ──────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run with sudo." >&2
  exit 1
fi

if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker not found. Install Docker first:" >&2
  echo "  curl -fsSL https://get.docker.com | sh" >&2
  exit 1
fi

if [[ ! -f "$SERVICE_FILE" ]]; then
  echo "ERROR: $SERVICE_FILE not found. Run from the repo root." >&2
  exit 1
fi

# ── Copy repo to install location ─────────────────────────────────────────────
echo "→ Copying repo to ${INSTALL_DIR}..."
mkdir -p "$INSTALL_DIR"
rsync -a --exclude='.git' --exclude='data/raw/*' --exclude='*.pyc' \
  "$(pwd)/" "$INSTALL_DIR/"

# ── Create .env if missing ────────────────────────────────────────────────────
if [[ ! -f "${INSTALL_DIR}/.env" ]]; then
  echo "→ Creating .env from .env.example — review and edit as needed."
  cp "${INSTALL_DIR}/.env.example" "${INSTALL_DIR}/.env"
fi

# ── Install and enable systemd service ────────────────────────────────────────
echo "→ Installing systemd service..."
sed "s|%E/INSTALL_DIR|${INSTALL_DIR}|g" "$SERVICE_FILE" \
  > "${SYSTEMD_DIR}/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl start  "${SERVICE_NAME}.service"

echo ""
echo "✓ Smart-Lite Insight installed and started."
echo ""
echo "Useful commands:"
echo "  systemctl status  ${SERVICE_NAME}"
echo "  systemctl restart ${SERVICE_NAME}"
echo "  systemctl stop    ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "Services:"
echo "  API:       http://$(hostname -I | awk '{print $1}'):8000"
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8501"
