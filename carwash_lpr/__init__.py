"""Licence plate recognition for car wash bays (Raspberry Pi, Moldovan plates)."""

import os

# ONNX Runtime starts a telemetry client that uploads usage data to Microsoft as soon as it
# is imported. A car wash camera box should not do that; this must be set before the import.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "1.0.0"
