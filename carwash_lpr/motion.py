"""Cheap frame differencing so the Pi only runs the neural networks when something moves."""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, width: int = 160, pixel_delta: int = 25):
        self.width = width
        self.pixel_delta = pixel_delta
        self._previous: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> float:
        """Fraction of pixels that changed since the previous call (1.0 for the first frame)."""
        h, w = frame.shape[:2]
        height = max(1, round(h * self.width / w))
        small = cv2.resize(frame, (self.width, height), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        previous, self._previous = self._previous, small
        if previous is None or previous.shape != small.shape:
            return 1.0
        diff = cv2.absdiff(small, previous)
        return float(np.count_nonzero(diff > self.pixel_delta)) / diff.size

    def reset(self) -> None:
        self._previous = None
