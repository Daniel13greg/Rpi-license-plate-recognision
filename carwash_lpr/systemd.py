"""Minimal sd_notify client (readiness and watchdog) without extra dependencies."""

from __future__ import annotations

import os
import socket


def notify(message: str) -> bool:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())
    except OSError:
        return False
    return True


def watchdog_interval() -> float | None:
    """Seconds between watchdog pings (half of WatchdogSec), or None if not enabled."""
    usec = os.environ.get("WATCHDOG_USEC")
    pid = os.environ.get("WATCHDOG_PID")
    if not usec or (pid and pid.isdigit() and int(pid) != os.getpid()):
        return None
    return int(usec) / 2_000_000
