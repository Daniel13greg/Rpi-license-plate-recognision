"""Events produced by the bays and the JSON sent to the car wash system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

SCHEMA = "carwash-lpr/1"
EVENT_TYPES = ("plate_recognized", "plate_unrecognized", "vehicle_left", "test")


def iso_time(timestamp: float) -> str:
    """Local time with UTC offset, e.g. 2026-10-05T14:03:07.123+03:00."""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")


@dataclass
class PlateEvent:
    event_type: str
    bay_id: str
    bay_name: str = ""
    trigger: str = "continuous"  # continuous | gpio | http | test
    plate: str | None = None
    plate_display: str | None = None
    plate_format: str | None = None
    country: str | None = None
    confidence: float | None = None
    votes: int = 0
    candidates: list[dict] = field(default_factory=list)
    ocr_region: str | None = None
    duration_seconds: float | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    frame: np.ndarray | None = field(default=None, repr=False, compare=False)
    crop: np.ndarray | None = field(default=None, repr=False, compare=False)
    frame_path: str | None = None
    plate_path: str | None = None

    def payload(self, device_id: str, site_id: str = "") -> dict:
        data = {
            "schema": SCHEMA,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": iso_time(self.created_at),
            "device_id": device_id,
            "site_id": site_id or None,
            "bay_id": self.bay_id,
            "bay_name": self.bay_name,
            "trigger": self.trigger,
            "plate": self.plate,
            "plate_display": self.plate_display,
            "plate_format": self.plate_format,
            "country": self.country,
            "confidence": None if self.confidence is None else round(self.confidence, 3),
            "votes": self.votes,
            "candidates": self.candidates,
            "ocr_region": self.ocr_region,
        }
        if self.duration_seconds is not None:
            data["duration_seconds"] = round(self.duration_seconds, 1)
        return data
