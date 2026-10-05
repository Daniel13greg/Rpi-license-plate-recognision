from __future__ import annotations

import numpy as np
import pytest

from carwash_lpr.config import BayConfig, Config, config_from_dict
from carwash_lpr.recognizer import PlateRead


def make_read(text: str, confidence: float = 0.95, box=(500, 500, 760, 560), region: str | None = "Moldova") -> PlateRead:
    x1, y1, x2, y2 = box
    return PlateRead(
        text=text,
        confidence=confidence,
        min_char_confidence=confidence,
        box=box,
        detection_confidence=0.8,
        region=region,
        region_confidence=0.9,
        crop=np.zeros((y2 - y1, x2 - x1, 3), np.uint8),
    )


class FakeRecognizer:
    """Returns whatever the test put in ``reads`` (a list, or a callable(frame) -> list)."""

    def __init__(self, reads=None):
        self.reads = reads if reads is not None else []
        self.calls = 0

    def recognize(self, frame, roi=None):
        self.calls += 1
        reads = self.reads(frame) if callable(self.reads) else self.reads
        return list(reads)


class FakeGrabber:
    status = "ok"
    last_error = ""

    def __init__(self, frame: np.ndarray | None = None):
        self.frame = frame if frame is not None else np.zeros((720, 1280, 3), np.uint8)
        self.seq = 0
        self.started = False

    def next_frame(self, frame: np.ndarray | None = None) -> None:
        if frame is not None:
            self.frame = frame
        self.seq += 1

    def latest(self):
        return (self.frame if self.seq else None), 0.0, self.seq

    def frame_age(self):
        return 0.0 if self.seq else None

    def start(self):
        self.started = True

    def stop(self, timeout: float = 5.0):
        self.started = False


class FakeSensor:
    def __init__(self):
        self.active = False
        self.closed = False

    def close(self):
        self.closed = True


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def bay_config(**overrides) -> dict:
    bay = {"id": "1", "camera": {"type": "images", "path": "/tmp"}, "motion": {"enabled": False}}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(bay.get(key), dict):
            bay[key] = {**bay[key], **value}
        else:
            bay[key] = value
    return bay


def make_config(tmp_path, bays: list[dict] | None = None, **top) -> Config:
    data = {"data_dir": str(tmp_path / "data"), "bays": bays or [bay_config()], **top}
    return config_from_dict(data, base_dir=tmp_path, environ={})


@pytest.fixture
def clock():
    return Clock()
