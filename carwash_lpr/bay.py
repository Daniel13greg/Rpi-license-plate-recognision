"""Per-bay logic: decides when a car has arrived and which plate it has.

Two modes:

continuous  No sensor. The camera is watched all the time (gated by motion); a car has
            arrived when its plate is confirmed by several frames and has left when the
            plate has not been seen for ``absence_timeout_seconds``.
trigger     A presence sensor on a GPIO pin (or an HTTP call) opens a read window; the
            first confirmed plate is reported, or ``plate_unrecognized`` when the window
            ends without one. The sensor releasing means the car has left.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from carwash_lpr import plates
from carwash_lpr.camera import FrameGrabber
from carwash_lpr.config import BayConfig, Config, PresenceConfig, TriggerConfig, VotingConfig
from carwash_lpr.events import PlateEvent
from carwash_lpr.motion import MotionDetector
from carwash_lpr.recognizer import PlateRead, Recognizer, draw_overlay, roi_pixels
from carwash_lpr.voting import Candidate, Observation, PlateVoter, observe, primary

log = logging.getLogger(__name__)

# In continuous mode a read request is answered at once if the current car's plate was
# seen this recently.
FRESH_SECONDS = 10.0


def same_vehicle(a: str, b: str) -> bool:
    """Plates one character apart are taken to be misreads of the same plate."""
    return plates.edit_distance(a, b) <= 1


@dataclass
class Outcome:
    kind: str  # arrived | confirmed | left | unrecognized
    candidate: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    source: str = "continuous"
    duration: float | None = None


class ContinuousTracker:
    mode = "continuous"

    def __init__(self, presence: PresenceConfig, voting: VotingConfig):
        self.cfg = presence
        self.voter = PlateVoter(voting)
        self.current: Candidate | None = None
        self.arrived_at = 0.0
        self.last_seen = 0.0
        self.reported = False  # whether the current car's arrival was reported
        self._recent: dict[str, float] = {}  # plate -> last time it was seen

    def frame_policy(self) -> str:
        return "motion"

    def update(self, observation: Observation | None, now: float) -> list[Outcome]:
        outcomes: list[Outcome] = []
        if observation is not None:
            self.voter.add(observation)
            if self.current is not None and same_vehicle(observation.plate, self.current.plate):
                self.last_seen = now
                self._recent[self.current.plate] = now
        self.voter.prune(now)
        candidates = self.voter.tally()
        decision = self.voter.decide(candidates)
        if decision is not None:
            self.voter.clear()
            if self.current is not None and same_vehicle(decision.plate, self.current.plate):
                outcomes.append(Outcome("confirmed", self.current))
            else:
                outcomes += self._departed()
                previous = self._recent.get(decision.plate)
                self.current, self.arrived_at, self.last_seen = decision, now, now
                self._recent[decision.plate] = now
                self.reported = previous is None or now - previous >= self.cfg.repeat_cooldown_seconds
                if self.reported:
                    outcomes.append(Outcome("arrived", decision, candidates))
                else:
                    log.info("%s seen again %.0fs after last time; not reported again", decision.plate, now - previous)
                    outcomes.append(Outcome("confirmed", decision))
        elif self.current is not None and now - self.last_seen > self.cfg.absence_timeout_seconds:
            outcomes += self._departed()
        for plate, seen in list(self._recent.items()):
            if now - seen > self.cfg.repeat_cooldown_seconds:
                del self._recent[plate]
        return outcomes

    def _departed(self) -> list[Outcome]:
        """Forget the current car; a departure is only reported if its arrival was."""
        if self.current is None:
            return []
        outcome = Outcome("left", self.current, duration=self.last_seen - self.arrived_at)
        self.current = None
        return [outcome] if self.reported else []

    def state(self) -> str:
        return "occupied" if self.current is not None else "empty"


class TriggerTracker:
    mode = "trigger"

    def __init__(self, trigger: TriggerConfig, voting: VotingConfig):
        self.cfg = trigger
        self.voter = PlateVoter(voting)
        self.phase = "idle"  # idle | reading | done
        self.source = ""
        self.started_at = 0.0
        self.result: Candidate | None = None
        self.sensor_active = False
        self.vehicle_present = False  # the sensor has seen a car that has not left yet
        self.present_since = 0.0

    def frame_policy(self) -> str:
        if self.phase == "reading":
            return "always"
        if self.phase == "idle" and self.cfg.pre_trigger_seconds > 0:
            return "motion"
        return "never"

    def start(self, source: str, now: float) -> bool:
        """Open a read window. False if one is open or the car present was already read."""
        if self.phase == "reading":
            return False
        if self.phase == "done" and self.sensor_active and self.result is not None:
            return False
        self.phase, self.source, self.started_at, self.result = "reading", source, now, None
        self.voter.discard_before(now - self.cfg.pre_trigger_seconds)
        return True

    def sensor_on(self, now: float) -> bool:
        """The presence sensor sees a car; returns True if a read window was opened."""
        self.sensor_active = True
        self.vehicle_present = True
        self.present_since = now
        return self.start("gpio", now)

    def release(self, now: float) -> list[Outcome]:
        """The presence sensor no longer sees a car."""
        self.sensor_active = False
        outcomes = []
        if self.phase == "reading":
            outcomes += self._give_up()
        if self.vehicle_present:
            outcomes.append(Outcome("left", self.result, source="gpio", duration=now - self.present_since))
        self.phase, self.result, self.vehicle_present = "idle", None, False
        self.voter.clear()
        return outcomes

    def update(self, observation: Observation | None, now: float) -> list[Outcome]:
        if observation is not None:
            self.voter.add(observation)
        if self.phase != "reading":
            # outside a read window only the last few seconds are kept, for the next trigger
            self.voter.discard_before(now - self.cfg.pre_trigger_seconds)
            return []
        candidates = self.voter.tally()
        decision = self.voter.decide(candidates)
        if decision is not None:
            self.result = decision
            self.phase = "done" if self.sensor_active else "idle"
            self.voter.clear()
            return [Outcome("arrived", decision, candidates, source=self.source)]
        if now - self.started_at >= self.cfg.window_seconds:
            return self._give_up()
        return []

    def _give_up(self) -> list[Outcome]:
        candidates = self.voter.tally()
        self.phase = "done" if self.sensor_active else "idle"
        self.voter.clear()
        return [Outcome("unrecognized", None, candidates, source=self.source)]

    def state(self) -> str:
        return self.phase


class Sensor(Protocol):
    @property
    def active(self) -> bool: ...

    def close(self) -> None: ...


class ReadRequest:
    """An HTTP caller waiting for the plate of the car in a bay."""

    def __init__(self, created: float):
        self.created = created
        self.result: dict | None = None
        self._done = threading.Event()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def resolve(self, result: dict) -> None:
        if not self._done.is_set():
            self.result = result
            self._done.set()

    def wait(self, timeout: float) -> dict | None:
        self._done.wait(timeout)
        return self.result


class BayWorker:
    def __init__(
        self,
        bay: BayConfig,
        cfg: Config,
        recognizer: Recognizer,
        grabber: FrameGrabber,
        emit: Callable[[PlateEvent], None],
        sensor: Sensor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.bay = bay
        self.cfg = cfg
        self.recognizer = recognizer
        self.grabber = grabber
        self.emit = emit
        self.sensor = sensor
        self.clock = clock
        if bay.mode == "continuous":
            self.tracker: ContinuousTracker | TriggerTracker = ContinuousTracker(bay.presence, cfg.voting)
        else:
            self.tracker = TriggerTracker(bay.trigger, cfg.voting)
        self.motion = MotionDetector() if bay.motion.enabled else None
        self.heartbeat = clock()
        self.last_motion = float("-inf")
        self.last_recognition = float("-inf")
        self.last_reads: list[PlateRead] = []
        self.last_reads_at = float("-inf")
        self.last_event: dict | None = None
        self.stats = {"frames": 0, "recognitions": 0, "reads": 0, "events": 0}
        self._last_seq = 0
        self._raw_sensor = False
        self._raw_since = clock()
        self._requests: list[ReadRequest] = []
        self._triggers: deque[str] = deque()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"bay-{bay.id}", daemon=True)

    # --- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        self.grabber.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(10)
        self.grabber.stop()
        if self.sensor is not None:
            self.sensor.close()

    def alive(self, max_silence: float = 30.0) -> bool:
        return self._thread.is_alive() and self.clock() - self.heartbeat < max_silence

    def _run(self) -> None:
        interval = 1.0 / self.bay.process_fps
        next_step = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            if now < next_step:
                self._stop.wait(min(next_step - now, 0.05))
                continue
            next_step = max(next_step + interval, now)
            try:
                self.step(now)
            except Exception:
                log.exception("bay %s: processing error", self.bay.id)
                self._stop.wait(1.0)

    # --- requests from other threads ---------------------------------------------

    def request_read(self, source: str = "http") -> ReadRequest:
        """Ask for the plate of the car in the bay (starts a read window in trigger mode)."""
        request = ReadRequest(self.clock())
        with self._lock:
            self._requests.append(request)
            self._triggers.append(source)
        return request

    # --- processing ------------------------------------------------------------------

    def step(self, now: float) -> None:
        """One processing cycle; called by the bay thread (or directly by tests)."""
        self.heartbeat = now
        with self._lock:
            outcomes = self._poll_sensor(now)
            while self._triggers:
                outcomes += self._start_read(self._triggers.popleft(), now)
            frame, _, seq = self.grabber.latest()
            fresh = frame is not None and seq != self._last_seq
            if fresh:
                self._last_seq = seq
                self.stats["frames"] += 1
            run = fresh and self._should_recognize(frame, now)
        observation = self._recognize(frame, now) if run else None
        with self._lock:
            outcomes += self.tracker.update(observation, now)
            for outcome in outcomes:
                self._handle(outcome, frame)
            self._requests = [r for r in self._requests if not r.done and now - r.created < 120]

    def _poll_sensor(self, now: float) -> list[Outcome]:
        if self.sensor is None or not isinstance(self.tracker, TriggerTracker):
            return []
        raw = bool(self.sensor.active)
        if raw != self._raw_sensor:
            self._raw_sensor, self._raw_since = raw, now
        held = now - self._raw_since
        trig = self.bay.trigger
        if raw and not self.tracker.sensor_active and held >= trig.activate_delay_seconds:
            log.info("bay %s: vehicle detected by sensor", self.bay.id)
            if self.tracker.sensor_on(now):
                log.info("bay %s: reading plate (gpio trigger)", self.bay.id)
            return []
        if not raw and self.tracker.sensor_active and held >= trig.release_delay_seconds:
            log.info("bay %s: sensor released", self.bay.id)
            return self.tracker.release(now)
        return []

    def _start_read(self, source: str, now: float) -> list[Outcome]:
        tracker = self.tracker
        if isinstance(tracker, TriggerTracker):
            if tracker.start(source, now):
                log.info("bay %s: reading plate (%s trigger)", self.bay.id, source)
            elif tracker.phase == "done" and tracker.result is not None:
                return [Outcome("confirmed", tracker.result, source=source)]
            return []
        if tracker.current is not None and now - tracker.last_seen <= FRESH_SECONDS:
            return [Outcome("confirmed", tracker.current, source=source)]
        return []

    def _should_recognize(self, frame: np.ndarray, now: float) -> bool:
        policy = self.tracker.frame_policy()
        if policy == "never":
            return False
        if policy == "always" or self.motion is None:
            return True
        x1, y1, x2, y2 = roi_pixels(self.bay.roi, frame.shape[1], frame.shape[0])
        if self.motion.update(frame[y1:y2, x1:x2]) >= self.bay.motion.threshold:
            self.last_motion = now
        if now - self.last_motion <= self.bay.motion.hold_seconds:
            return True
        idle = self.bay.motion.idle_interval_seconds
        return idle > 0 and now - self.last_recognition >= idle

    def _recognize(self, frame: np.ndarray, now: float) -> Observation | None:
        self.last_recognition = now
        roi = roi_pixels(self.bay.roi, frame.shape[1], frame.shape[0])
        try:
            reads = self.recognizer.recognize(frame, roi)
        except Exception:
            log.exception("bay %s: recognition failed", self.bay.id)
            return None
        observations = [o for o in (observe(r, self.cfg.plates, now, frame) for r in reads) if o is not None]
        with self._lock:
            self.stats["recognitions"] += 1
            self.stats["reads"] += len(reads)
            self.last_reads, self.last_reads_at = reads, now
        if reads:
            log.debug(
                "bay %s: %s",
                self.bay.id,
                ", ".join(f"{r.text} {r.confidence:.2f} {r.width}px" for r in reads),
            )
        return primary(observations)

    def _handle(self, outcome: Outcome, frame: np.ndarray | None) -> None:
        event = None
        if outcome.kind == "arrived":
            event = self._event("plate_recognized", outcome, frame)
        elif outcome.kind == "unrecognized" and self.bay.trigger.report_unrecognized:
            event = self._event("plate_unrecognized", outcome, frame)
        elif outcome.kind == "left":
            if isinstance(self.tracker, ContinuousTracker):
                report = self.bay.presence.report_departures
            else:
                report = self.bay.trigger.report_departures
            if report:
                event = self._event("vehicle_left", outcome, None)
        if outcome.kind in ("arrived", "confirmed", "unrecognized"):
            result = self._result(outcome, event)
            for request in self._requests:
                request.resolve(result)
        if outcome.kind == "arrived":
            log.info(
                "bay %s: plate %s (%s, %.0f%%, %d reads)",
                self.bay.id, outcome.candidate.display, outcome.candidate.format,
                outcome.candidate.confidence * 100, outcome.candidate.votes,
            )
        elif outcome.kind == "unrecognized":
            log.info("bay %s: no plate recognised", self.bay.id)
        elif outcome.kind == "left":
            log.info("bay %s: vehicle left (%s)", self.bay.id, outcome.candidate.plate if outcome.candidate else "unknown plate")
        if event is not None:
            self.stats["events"] += 1
            self.emit(event)
            self.last_event = {k: v for k, v in event.payload("").items() if k in _LAST_EVENT_KEYS}

    def _event(self, event_type: str, outcome: Outcome, frame: np.ndarray | None) -> PlateEvent:
        cand = outcome.candidate
        best = cand.best if cand is not None else None
        read = best.read if best is not None else None
        return PlateEvent(
            event_type=event_type,
            bay_id=self.bay.id,
            bay_name=self.bay.name,
            trigger=outcome.source,
            plate=cand.plate if cand else None,
            plate_display=cand.display if cand else None,
            plate_format=cand.format if cand else None,
            country=cand.country if cand else None,
            confidence=cand.confidence if cand else None,
            votes=cand.votes if cand else 0,
            candidates=[c.summary() for c in outcome.candidates[:5]],
            ocr_region=read.region if read else None,
            duration_seconds=outcome.duration,
            frame=best.frame if best is not None and best.frame is not None else frame,
            crop=read.crop if read is not None else None,
        )

    def _result(self, outcome: Outcome, event: PlateEvent | None) -> dict:
        cand = outcome.candidate
        return {
            "bay_id": self.bay.id,
            "status": "recognized" if cand is not None else "unrecognized",
            "plate": cand.plate if cand else None,
            "plate_display": cand.display if cand else None,
            "confidence": round(cand.confidence, 3) if cand else None,
            "event_id": event.event_id if event is not None else None,
            "candidates": [c.summary() for c in outcome.candidates[:5]],
        }

    # --- introspection ---------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            tracker = self.tracker
            current = tracker.current if isinstance(tracker, ContinuousTracker) else tracker.result
            age = self.grabber.frame_age()
            now = self.clock()
            return {
                "id": self.bay.id,
                "name": self.bay.name,
                "mode": self.bay.mode,
                "state": tracker.state(),
                "plate": current.plate if current else None,
                "plate_display": current.display if current else None,
                "sensor_active": tracker.sensor_active if isinstance(tracker, TriggerTracker) else None,
                "camera": {
                    "status": self.grabber.status,
                    "error": self.grabber.last_error or None,
                    "frame_age_seconds": None if age is None else round(age, 2),
                },
                "last_reads": [
                    {"text": r.text, "confidence": round(r.confidence, 3), "width_px": r.width, "region": r.region}
                    for r in self.last_reads
                ] if now - self.last_reads_at < 10 else [],
                "last_event": self.last_event,
                "stats": dict(self.stats),
            }

    def snapshot(self, annotate: bool = True) -> np.ndarray | None:
        frame, _, _ = self.grabber.latest()
        if frame is None or not annotate:
            return frame
        with self._lock:
            reads = self.last_reads if self.clock() - self.last_reads_at < 5 else []
        roi = roi_pixels(self.bay.roi, frame.shape[1], frame.shape[0])
        labels = []
        for r in reads:
            found = plates.interpret(r.text)
            labels.append(found.display if found else r.text)
        return draw_overlay(frame, roi, reads, labels)


_LAST_EVENT_KEYS = ("event_id", "event_type", "timestamp", "plate", "plate_display", "confidence", "trigger")
