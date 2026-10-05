#!/usr/bin/env bash
# Installs (or updates) carwash-lpr on Raspberry Pi OS 64-bit (Bookworm or newer).
#
#   git clone https://github.com/Daniel13greg/Rpi-license-plate-recognision.git
#   cd Rpi-license-plate-recognision && sudo ./deploy/install.sh
#
# Re-running it updates the program and keeps the configuration and data.
set -euo pipefail

APP_DIR=/opt/carwash-lpr
CONF_DIR=/etc/carwash-lpr
DATA_DIR=/var/lib/carwash-lpr
SERVICE_USER=carwash-lpr
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root: sudo $0"
arch="$(uname -m)"
if [[ $arch != aarch64 && $arch != x86_64 ]]; then
    die "a 64-bit OS is required (found $arch). Install Raspberry Pi OS (64-bit)."
fi
command -v apt-get >/dev/null || die "this installer needs a Debian based system (Raspberry Pi OS)"

say "Installing system packages"
apt-get update
packages=(python3 python3-venv python3-pip python3-numpy python3-yaml python3-requests sqlite3 curl)
# Raspberry Pi specific packages: camera stack and GPIO (absent on other systems).
for pkg in python3-picamera2 rpicam-apps python3-gpiozero python3-lgpio; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then packages+=("$pkg"); fi
done
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"

say "Creating service user $SERVICE_USER"
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
for group in video gpio render; do
    if getent group "$group" >/dev/null; then usermod -aG "$group" "$SERVICE_USER"; fi
done
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 750 "$DATA_DIR"

say "Installing the program into $APP_DIR"
# The interpreter apt's Python packages (picamera2, gpiozero, numpy) are built for.
PYTHON=/usr/bin/python3
default_python="$(sed -n 's/^default-version = //p' /usr/share/python3/debian_defaults 2>/dev/null || true)"
if [[ -n $default_python && -x /usr/bin/$default_python ]]; then
    PYTHON="/usr/bin/$default_python"
fi
version_of() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'; }
install -d -m 755 "$APP_DIR"
if [[ -x $APP_DIR/venv/bin/python && "$(version_of "$APP_DIR/venv/bin/python")" != "$(version_of "$PYTHON")" ]]; then
    echo "Python changed: recreating the virtualenv"
    rm -rf "$APP_DIR/venv"
fi
# --system-site-packages: picamera2, libcamera and gpiozero come from apt.
if [[ ! -x $APP_DIR/venv/bin/python ]]; then
    "$PYTHON" -m venv --system-site-packages "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
# Keep the system's numpy: picamera2 and simplejpeg are compiled against it, and a
# different numpy in the virtualenv breaks "import picamera2".
constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
if sys_numpy="$("$PYTHON" -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"; then
    echo "numpy==$sys_numpy" >"$constraints"
    echo "keeping the system numpy $sys_numpy"
else
    echo "WARNING: cannot import the system numpy with $PYTHON; pip may install another" >&2
    echo "         numpy version, which can break picamera2 (camera type 'rpicam' still works)." >&2
fi
"$APP_DIR/venv/bin/pip" install --quiet --constraint "$constraints" "$SRC_DIR"
ln -sf "$APP_DIR/venv/bin/carwash-lpr" /usr/local/bin/carwash-lpr
if dpkg -s python3-picamera2 >/dev/null 2>&1; then
    if "$APP_DIR/venv/bin/python" -c "import picamera2" 2>/dev/null; then
        echo "picamera2 works inside the virtualenv"
    else
        echo "WARNING: 'import picamera2' fails inside the virtualenv; use camera type 'rpicam'" >&2
    fi
fi

say "Configuration in $CONF_DIR"
install -d -m 755 "$CONF_DIR"
if [[ ! -f $CONF_DIR/config.yaml ]]; then
    sed "s/^device_id: .*/device_id: $(hostname)/" "$SRC_DIR/config/config.example.yaml" >"$CONF_DIR/config.yaml"
    chmod 644 "$CONF_DIR/config.yaml"
    echo "created $CONF_DIR/config.yaml"
else
    echo "keeping existing $CONF_DIR/config.yaml"
fi
if [[ ! -f $CONF_DIR/env ]]; then
    api_token="$("$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))')"
    cat >"$CONF_DIR/env" <<EOF
# Secrets for /etc/carwash-lpr/config.yaml (referenced there as \${NAME}).
CARWASH_WEBHOOK_URL=
CARWASH_API_TOKEN=
CARWASH_HMAC_SECRET=
LPR_API_TOKEN=$api_token
EOF
    echo "created $CONF_DIR/env with a random API token"
fi
chown root:"$SERVICE_USER" "$CONF_DIR/env"
chmod 640 "$CONF_DIR/env"

say "Downloading the recognition models"
sudo -u "$SERVICE_USER" env HOME="$DATA_DIR" "$APP_DIR/venv/bin/carwash-lpr" download-models -c "$CONF_DIR/config.yaml"

say "Installing the systemd service"
install -m 644 "$SRC_DIR/deploy/carwash-lpr.service" /etc/systemd/system/carwash-lpr.service
systemctl daemon-reload
systemctl enable carwash-lpr.service >/dev/null
systemctl restart carwash-lpr.service

ip_address="$(hostname -I 2>/dev/null | awk '{print $1}')"
token="$(sed -n 's/^LPR_API_TOKEN=//p' "$CONF_DIR/env")"
cat <<EOF

Installed. Next steps:
  1. Set the car wash system's address and secrets:   sudo nano $CONF_DIR/env
  2. Adjust bays, cameras and options:                 sudo nano $CONF_DIR/config.yaml
  3. Check and apply:   carwash-lpr check-config && sudo systemctl restart carwash-lpr
  4. Aim the camera:    http://${ip_address:-<pi-address>}:8080/?token=$token
  5. Logs:              journalctl -u carwash-lpr -f
EOF
