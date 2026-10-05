"""Delivers queued events to the car wash system over HTTP(S).

Each request carries ``Idempotency-Key`` / ``X-LPR-Event-Id`` (the event id, so the
server can ignore retried duplicates), an optional bearer token and an optional
HMAC-SHA256 signature: ``X-LPR-Signature: sha256=<hex>`` computed over
``<X-LPR-Timestamp>.<raw body>`` with the shared secret.

Failed deliveries are retried with exponential backoff until the event is older than
``max_event_age_seconds``: a "car entered bay 2" message delivered ten minutes late
could start a wash session for the wrong car, so stale events are dropped instead.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests

from carwash_lpr import __version__
from carwash_lpr.config import ConfigError, WebhookConfig
from carwash_lpr.events import PlateEvent
from carwash_lpr.outbox import Outbox, QueuedEvent

log = logging.getLogger(__name__)

_WHOLE_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_RETRYABLE = {408, 425, 429}


class _Values(dict):
    """format_map() mapping: unknown names raise KeyError, None renders as ''."""

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        return "" if value is None else value


def render_template(template: Any, values: dict) -> Any:
    """Fill ``{placeholders}`` in a JSON-like template.

    A string that is exactly one placeholder keeps the value's type (number, null, list),
    so ``{"licensePlate": "{plate}", "box": "{bay_id}"}`` produces proper JSON values.
    """
    if isinstance(template, dict):
        return {str(k): render_template(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, values) for v in template]
    if isinstance(template, str):
        whole = _WHOLE_PLACEHOLDER.match(template)
        if whole:
            return values[whole.group(1)]
        return template.format_map(_Values(values))
    return template


def sign(secret: str, timestamp: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _sample_payload() -> dict:
    return PlateEvent("plate_recognized", "1", "Bay 1", plate="ABC123", plate_display="ABC 123").payload("pi", "site")


def check_templates(cfg: WebhookConfig) -> None:
    """Fail at start-up, not at the first car, if a template uses an unknown placeholder."""
    values = _sample_payload()
    try:
        if cfg.payload_template is not None:
            render_template(cfg.payload_template, values)
        cfg.url.format_map(_Values(values))
    except KeyError as exc:
        raise ConfigError(
            f"webhook: unknown placeholder {{{exc.args[0]}}}; available: {', '.join(sorted(values))}"
        ) from None
    except (ValueError, IndexError) as exc:
        raise ConfigError(f"webhook: bad template: {exc}") from None


class WebhookSender:
    def __init__(
        self,
        cfg: WebhookConfig,
        outbox: Outbox,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.time,
    ):
        check_templates(cfg)
        self.cfg = cfg
        self.outbox = outbox
        self.session = session or requests.Session()
        self.clock = clock
        self.heartbeat = time.monotonic()
        self.last_error: str | None = None
        self.last_delivery_at: float | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="webhook", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(self.cfg.timeout_seconds + 5)

    def notify(self) -> None:
        """A new event was queued."""
        self._wake.set()

    def alive(self, max_silence: float = 120.0) -> bool:
        return self._thread.is_alive() and time.monotonic() - self.heartbeat < max_silence

    def _run(self) -> None:
        while not self._stop.is_set():
            self.heartbeat = time.monotonic()
            try:
                self.deliver_due()
            except Exception:
                log.exception("webhook sender error")
            next_at = self.outbox.next_attempt_at()
            wait = 5.0 if next_at is None else min(max(next_at - self.clock(), 0.05), 5.0)
            self._wake.wait(wait)
            self._wake.clear()

    def deliver_due(self) -> int:
        """Send everything that is due now; returns the number of attempts made."""
        attempts = 0
        for _ in range(100):  # bounded so the heartbeat keeps beating during long outages
            batch = self.outbox.due(self.clock())
            if not batch or self._stop.is_set():
                break
            for event in batch:
                self._deliver(event)
                attempts += 1
        return attempts

    def build_request(self, event: QueuedEvent) -> tuple[str, bytes, dict[str, str]]:
        payload = dict(event.payload)
        if self.cfg.include_images:
            payload["images"] = {
                "plate_jpeg_base64": _read_base64(event.plate_path),
                "frame_jpeg_base64": _read_base64(event.frame_path),
            }
        if self.cfg.payload_template is not None:
            payload = render_template(self.cfg.payload_template, payload)
        url = self.cfg.url.format_map(_Values(event.payload)) if "{" in self.cfg.url else self.cfg.url
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": f"carwash-lpr/{__version__}",
            "Idempotency-Key": event.event_id,
            "X-LPR-Event-Id": event.event_id,
        }
        if self.cfg.bearer_token:
            headers["Authorization"] = f"Bearer {self.cfg.bearer_token}"
        if self.cfg.hmac_secret:
            timestamp = str(int(self.clock()))
            headers["X-LPR-Timestamp"] = timestamp
            headers["X-LPR-Signature"] = sign(self.cfg.hmac_secret, timestamp, body)
        headers.update(self.cfg.headers)
        return url, body, headers

    def _deliver(self, event: QueuedEvent) -> None:
        now = self.clock()
        if now - event.created_at > self.cfg.max_event_age_seconds:
            reason = f"not delivered within {self.cfg.max_event_age_seconds:.0f}s"
            self.outbox.mark_expired(event.seq, reason)
            log.warning("dropping %s for bay %s (%s): %s", event.event_type, event.bay_id, event.event_id, reason)
            return
        url, body, headers = self.build_request(event)
        verify: bool | str = self.cfg.ca_bundle or self.cfg.verify_tls
        try:
            response = self.session.request(
                self.cfg.method, url, data=body, headers=headers, timeout=self.cfg.timeout_seconds, verify=verify
            )
        except requests.RequestException as exc:
            self._retry(event, f"{type(exc).__name__}: {exc}")
            return
        code = response.status_code
        text = response.text[:2000]
        if 200 <= code < 300:
            self.outbox.mark_delivered(event.seq, self.clock(), code, text)
            self.last_error = None
            self.last_delivery_at = self.clock()
            log.info("delivered %s for bay %s (%s): HTTP %d", event.event_type, event.bay_id, event.event_id, code)
        elif code in _RETRYABLE or code >= 500:
            self._retry(event, f"HTTP {code}: {text[:200]}", code, response.headers.get("Retry-After"))
        else:
            self.outbox.mark_failed(event.seq, f"HTTP {code}", code, text)
            self.last_error = f"HTTP {code}: {text[:200]}"
            log.error(
                "car wash system rejected %s for bay %s (%s): HTTP %d %s",
                event.event_type, event.bay_id, event.event_id, code, text[:200],
            )

    def _retry(self, event: QueuedEvent, error: str, code: int | None = None, retry_after: str | None = None) -> None:
        delay = min(2.0**event.attempts, 30.0)
        if retry_after and retry_after.strip().isdigit():
            delay = min(max(delay, float(retry_after)), 60.0)
        self.outbox.mark_retry(event.seq, self.clock() + delay, error, code)
        self.last_error = error
        log.warning("delivery of %s for bay %s failed (%s); retrying in %.0fs", event.event_type, event.bay_id, error, delay)


def _read_base64(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return None
