"""Wires cameras, recognizer, bays, event storage, delivery and the API together."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path
from typing import Callable

from carwash_lpr import __version__, systemd
from carwash_lpr.api import ApiServer
from carwash_lpr.bay import BayWorker, Sensor
from carwash_lpr.camera import FrameGrabber, FrameSource, create_source
from carwash_lpr.config import CameraConfig, Config, TriggerConfig
from carwash_lpr.events import PlateEvent
from carwash_lpr.outbox import Outbox
from carwash_lpr.recognizer import Recognizer
from carwash_lpr.sender import WebhookSender
from carwash_lpr.storage import SnapshotStore

log = logging.getLogger(__name__)

# A camera that is connected but has produced no frame for this long is wedged; the
# service then stops answering the systemd watchdog so it gets restarted.
CAMERA_STALL_SECONDS = 60.0


def _gpio_sensor(cfg: TriggerConfig) -> Sensor:
    from carwash_lpr.gpio import GpioSensor

    return GpioSensor.from_config(cfg)


class Service:
    def __init__(
        self,
        cfg: Config,
        recognizer: Recognizer | None = None,
        source_factory: Callable[[CameraConfig], FrameSource] = create_source,
        sensor_factory: Callable[[TriggerConfig], Sensor] = _gpio_sensor,
    ):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outbox = Outbox(self.data_dir / "events.db")
        self.snapshots = SnapshotStore(self.data_dir / "snapshots", cfg.storage)
        if recognizer is None:
            from carwash_lpr.recognizer import PlateRecognizer

            recognizer = PlateRecognizer(cfg.recognizer, self.data_dir / "models")
        self.recognizer = recognizer
        self.sender = WebhookSender(cfg.webhook, self.outbox) if cfg.webhook.url else None
        if self.sender is None:
            log.warning("webhook.url is not set: events are only stored locally and served by the API")
        self.bays: dict[str, BayWorker] = {}
        for bay in cfg.bays:
            sensor = None
            if bay.mode == "trigger" and bay.trigger.gpio_pin is not None:
                sensor = sensor_factory(bay.trigger)
            elif bay.mode == "trigger":
                log.info("bay %s: trigger mode without gpio_pin, reads start from the API only", bay.id)
            grabber = FrameGrabber(source_factory(bay.camera), bay.id)
            self.bays[bay.id] = BayWorker(bay, cfg, recognizer, grabber, self.emit, sensor)
        self.api = ApiServer(cfg.api, self) if cfg.api.enabled else None
        self.started_at = time.time()
        self._emit_lock = threading.Lock()
        self._stop = threading.Event()

    # --- events --------------------------------------------------------------------

    def emit(self, event: PlateEvent) -> None:
        """Store an event (snapshot + outbox) and hand it to the sender."""
        with self._emit_lock:
            if self.cfg.storage.save_snapshots:
                try:
                    self.snapshots.save(event)
                except Exception:
                    log.exception("could not save snapshot for event %s", event.event_id)
            deliver = self.sender is not None and (
                event.event_type in self.cfg.webhook.event_types or event.event_type == "test"
            )
            payload = event.payload(self.cfg.device_id, self.cfg.site_id)
            self.outbox.add(event, payload, deliver)
        if deliver:
            self.sender.notify()

    # --- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        log.info("carwash-lpr %s starting on %s with %d bay(s)", __version__, self.cfg.device_id, len(self.bays))
        if self.sender is not None:
            self.sender.start()
        for worker in self.bays.values():
            worker.start()
        if self.api is not None:
            self.api.start()

    def stop(self) -> None:
        if self.api is not None:
            self.api.stop()
        for worker in self.bays.values():
            worker.stop()
        if self.sender is not None:
            self.sender.stop()
        self.outbox.close()
        log.info("stopped")

    def request_stop(self, *_args) -> None:
        self._stop.set()

    def run(self) -> None:
        """Run until SIGTERM/SIGINT, feeding the systemd watchdog while healthy."""
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.start()
        systemd.notify("READY=1")
        interval = systemd.watchdog_interval()
        next_housekeeping = 0.0
        try:
            while not self._stop.wait(min(interval or 5.0, 5.0)):
                problems = self.problems()
                if interval and not problems:
                    systemd.notify("WATCHDOG=1")
                elif problems:
                    log.warning("unhealthy: %s", "; ".join(problems))
                if time.monotonic() >= next_housekeeping:
                    self.housekeeping()
                    next_housekeeping = time.monotonic() + 3600
        finally:
            systemd.notify("STOPPING=1")
            self.stop()

    def housekeeping(self) -> None:
        try:
            self.snapshots.cleanup()
            purged = self.outbox.purge(time.time() - self.cfg.storage.event_retention_days * 86_400)
            if purged:
                log.info("purged %d old events", purged)
        except Exception:
            log.exception("housekeeping failed")

    # --- health ----------------------------------------------------------------------

    def problems(self) -> list[str]:
        """Faults the service cannot recover from by itself (they warrant a restart)."""
        found = []
        for worker in self.bays.values():
            if not worker.alive():
                found.append(f"bay {worker.bay.id} processing stalled")
            age = worker.grabber.frame_age()
            if worker.grabber.status == "ok" and age is not None and age > CAMERA_STALL_SECONDS:
                found.append(f"bay {worker.bay.id} camera stalled for {age:.0f}s")
        if self.sender is not None and not self.sender.alive():
            found.append("webhook sender stalled")
        return found

    def health(self) -> dict:
        problems = self.problems()
        cameras = {bay_id: w.grabber.status for bay_id, w in self.bays.items()}
        ok = not problems and all(status in ("ok", "ended") for status in cameras.values())
        return {"status": "ok" if ok else "degraded", "version": __version__, "cameras": cameras, "problems": problems}

    def status(self) -> dict:
        return {
            "version": __version__,
            "device_id": self.cfg.device_id,
            "site_id": self.cfg.site_id or None,
            "uptime_seconds": round(time.time() - self.started_at),
            "bays": [w.status() for w in self.bays.values()],
            "outbox": self.outbox.counts(),
            "webhook": {
                "enabled": self.sender is not None,
                "last_error": self.sender.last_error if self.sender else None,
                "last_delivery_at": self.sender.last_delivery_at if self.sender else None,
            },
            "problems": self.problems(),
        }
