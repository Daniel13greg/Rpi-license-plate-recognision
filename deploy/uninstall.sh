#!/usr/bin/env bash
# Removes carwash-lpr. Configuration and data are kept unless --purge is given.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0 [--purge]" >&2; exit 1; }

systemctl disable --now carwash-lpr.service 2>/dev/null || true
rm -f /etc/systemd/system/carwash-lpr.service /usr/local/bin/carwash-lpr
systemctl daemon-reload
rm -rf /opt/carwash-lpr

if [[ ${1:-} == --purge ]]; then
    rm -rf /etc/carwash-lpr /var/lib/carwash-lpr
    userdel carwash-lpr 2>/dev/null || true
    echo "removed program, configuration, events, snapshots and models"
else
    echo "removed program; kept /etc/carwash-lpr and /var/lib/carwash-lpr (use --purge to delete them)"
fi
