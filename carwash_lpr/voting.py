"""Combining reads from several frames into one trustworthy plate.

A single frame can be misread (water drops, motion blur, a reflection). The bay only
reports a plate after ``min_reads`` frames inside a short time window agree on it, it
holds most of the window's votes, and its average confidence is high enough.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from carwash_lpr import plates
from carwash_lpr.config import PlatesConfig, VotingConfig
from carwash_lpr.recognizer import PlateRead

# Confidence kept for every character that had to be changed to fit the plate layout.
FIX_CONFIDENCE = 0.8


@dataclass(frozen=True)
class Observation:
    """A plausible plate read from one frame."""

    plate: str
    display: str
    format: str
    country: str
    category: str
    confidence: float
    timestamp: float
    read: PlateRead | None = field(default=None, compare=False, repr=False)
    frame: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Candidate:
    """All observations of one plate in the voting window."""

    plate: str
    display: str
    format: str
    country: str
    votes: int
    score: float  # sum of the observations' confidences
    confidence: float  # average confidence
    best: Observation  # most confident observation (used for snapshots)

    def summary(self) -> dict:
        return {
            "plate": self.plate,
            "plate_display": self.display,
            "votes": self.votes,
            "confidence": round(self.confidence, 3),
        }


def observe(read: PlateRead, cfg: PlatesConfig, now: float, frame: np.ndarray | None = None) -> Observation | None:
    """Turn an OCR read into an Observation, or None if it is not an accepted plate."""
    found = plates.interpret(read.text)
    if found is None or found.category not in cfg.accept:
        return None
    confidence = read.confidence * FIX_CONFIDENCE**found.fixes
    if confidence < cfg.min_read_confidence:
        return None
    return Observation(
        plate=found.plate,
        display=found.display,
        format=found.format,
        country=found.country,
        category=found.category,
        confidence=confidence,
        timestamp=now,
        read=read,
        frame=frame,
    )


def primary(observations: Iterable[Observation]) -> Observation | None:
    """The plate of the vehicle closest to the camera: the biggest one in the frame."""
    return max(
        observations,
        key=lambda o: (o.read.area if o.read is not None else 0, o.confidence),
        default=None,
    )


class PlateVoter:
    def __init__(self, cfg: VotingConfig):
        self.cfg = cfg
        self._observations: deque[Observation] = deque()

    def __len__(self) -> int:
        return len(self._observations)

    def add(self, observation: Observation) -> None:
        self._observations.append(observation)

    def prune(self, now: float) -> None:
        self.discard_before(now - self.cfg.window_seconds)

    def discard_before(self, timestamp: float) -> None:
        while self._observations and self._observations[0].timestamp < timestamp:
            self._observations.popleft()

    def clear(self) -> None:
        self._observations.clear()

    def tally(self) -> list[Candidate]:
        groups: dict[str, list[Observation]] = {}
        for obs in self._observations:
            groups.setdefault(obs.plate, []).append(obs)
        result = []
        for plate, group in groups.items():
            score = sum(o.confidence for o in group)
            best = max(group, key=lambda o: o.confidence)
            result.append(
                Candidate(
                    plate=plate,
                    display=best.display,
                    format=best.format,
                    country=best.country,
                    votes=len(group),
                    score=score,
                    confidence=score / len(group),
                    best=best,
                )
            )
        result.sort(key=lambda c: (c.score, c.votes, c.best.timestamp), reverse=True)
        return result

    def decide(self, candidates: list[Candidate] | None = None) -> Candidate | None:
        """The winning plate if the evidence is strong enough, else None."""
        if candidates is None:
            candidates = self.tally()
        if not candidates:
            return None
        top = candidates[0]
        total = sum(c.score for c in candidates)
        if (
            top.votes >= self.cfg.min_reads
            and top.confidence >= self.cfg.min_confidence
            and top.score / total >= self.cfg.min_agreement
        ):
            return top
        return None
