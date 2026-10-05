import base64
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from carwash_lpr.config import ConfigError, WebhookConfig
from carwash_lpr.events import PlateEvent
from carwash_lpr.outbox import Outbox
from carwash_lpr.sender import WebhookSender, check_templates, render_template, sign


class Receiver:
    """A tiny HTTP server that records requests and answers with scripted status codes."""

    def __init__(self):
        self.requests = []
        self.responses = []  # (status, body, headers) popped per request; default 200
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                receiver.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body})
                status, text, headers = receiver.responses.pop(0) if receiver.responses else (200, '{"ok":true}', {})
                data = text.encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_POST = do_PUT = _handle

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver():
    r = Receiver()
    yield r
    r.close()


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def queue(tmp_path, created=1000.0, **event_fields):
    outbox = Outbox(tmp_path / "events.db")
    event = PlateEvent("plate_recognized", "2", "Box 2", plate="KCA123", plate_display="KCA 123", created_at=created, **event_fields)
    outbox.add(event, event.payload("pi-1", "site-a"), deliver=True)
    return outbox, event


def test_delivers_signed_request(tmp_path, receiver):
    outbox, event = queue(tmp_path)
    cfg = WebhookConfig(url=receiver.url + "/lpr", bearer_token="tok", hmac_secret="s3cret", headers={"X-Site": "a"})
    sender = WebhookSender(cfg, outbox, clock=Clock(1001.0))
    assert sender.deliver_due() == 1
    request = receiver.requests[0]
    headers = request["headers"]
    assert request["path"] == "/lpr"
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Idempotency-Key"] == event.event_id
    assert headers["X-Site"] == "a"
    expected = hmac.new(b"s3cret", headers["X-LPR-Timestamp"].encode() + b"." + request["body"], hashlib.sha256).hexdigest()
    assert headers["X-LPR-Signature"] == f"sha256={expected}"
    body = json.loads(request["body"])
    assert body["plate"] == "KCA123"
    assert body["bay_id"] == "2"
    assert body["device_id"] == "pi-1"
    row = outbox.recent(1)[0]
    assert row["status"] == "delivered"
    assert row["response_code"] == 200


def test_server_error_is_retried(tmp_path, receiver):
    outbox, _ = queue(tmp_path)
    clock = Clock(1001.0)
    sender = WebhookSender(WebhookConfig(url=receiver.url), outbox, clock=clock)
    receiver.responses.append((503, "busy", {}))
    sender.deliver_due()
    row = outbox.recent(1)[0]
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert sender.deliver_due() == 0  # backoff not over
    clock.now += 1.5
    sender.deliver_due()
    assert outbox.recent(1)[0]["status"] == "delivered"
    assert len(receiver.requests) == 2


def test_retry_after_is_respected(tmp_path, receiver):
    outbox, _ = queue(tmp_path)
    clock = Clock(1001.0)
    sender = WebhookSender(WebhookConfig(url=receiver.url), outbox, clock=clock)
    receiver.responses.append((429, "slow down", {"Retry-After": "20"}))
    sender.deliver_due()
    assert outbox.next_attempt_at() == pytest.approx(1021.0)


def test_client_error_is_final(tmp_path, receiver):
    outbox, _ = queue(tmp_path)
    sender = WebhookSender(WebhookConfig(url=receiver.url), outbox, clock=Clock(1001.0))
    receiver.responses.append((422, '{"error":"unknown bay"}', {}))
    sender.deliver_due()
    row = outbox.recent(1)[0]
    assert row["status"] == "failed"
    assert row["response_body"] == '{"error":"unknown bay"}'
    assert sender.last_error.startswith("HTTP 422")


def test_unreachable_server_is_retried(tmp_path):
    outbox, _ = queue(tmp_path)
    sender = WebhookSender(WebhookConfig(url="http://127.0.0.1:9", timeout_seconds=1), outbox, clock=Clock(1001.0))
    sender.deliver_due()
    row = outbox.recent(1)[0]
    assert row["status"] == "pending"
    assert "ConnectionError" in row["last_error"]


def test_stale_events_are_dropped(tmp_path, receiver):
    outbox, _ = queue(tmp_path)
    sender = WebhookSender(WebhookConfig(url=receiver.url, max_event_age_seconds=60), outbox, clock=Clock(1100.0))
    sender.deliver_due()
    assert receiver.requests == []
    assert outbox.recent(1)[0]["status"] == "expired"


def test_payload_template_and_url_placeholders(tmp_path, receiver):
    outbox, _ = queue(tmp_path)
    cfg = WebhookConfig(
        url=receiver.url + "/boxes/{bay_id}/car",
        method="PUT",
        payload_template={"licensePlate": "{plate}", "box": "{bay_id}", "note": "bay {bay_name}: {plate_display}", "score": "{confidence}"},
    )
    WebhookSender(cfg, outbox, clock=Clock(1001.0)).deliver_due()
    request = receiver.requests[0]
    assert request["method"] == "PUT"
    assert request["path"] == "/boxes/2/car"
    assert json.loads(request["body"]) == {"licensePlate": "KCA123", "box": "2", "note": "bay Box 2: KCA 123", "score": None}


def test_images_are_attached(tmp_path, receiver):
    plate_file = tmp_path / "plate.jpg"
    plate_file.write_bytes(b"\xff\xd8jpeg\xff\xd9")
    outbox, _ = queue(tmp_path, plate_path=str(plate_file))
    WebhookSender(WebhookConfig(url=receiver.url, include_images=True), outbox, clock=Clock(1001.0)).deliver_due()
    images = json.loads(receiver.requests[0]["body"])["images"]
    assert base64.b64decode(images["plate_jpeg_base64"]) == b"\xff\xd8jpeg\xff\xd9"
    assert images["frame_jpeg_base64"] is None


def test_template_errors_are_found_at_start():
    with pytest.raises(ConfigError, match="licence_plate"):
        check_templates(WebhookConfig(url="https://x", payload_template={"p": "{licence_plate}"}))
    with pytest.raises(ConfigError, match="bay"):
        check_templates(WebhookConfig(url="https://x/{bay}"))


def test_render_template_keeps_types():
    values = {"votes": 3, "plate": "KCA123", "candidates": [{"plate": "KCA123"}], "site_id": None}
    assert render_template({"n": "{votes}", "c": "{candidates}", "s": "{site_id}", "t": "plate={plate}", "k": 5}, values) == {
        "n": 3, "c": [{"plate": "KCA123"}], "s": None, "t": "plate=KCA123", "k": 5,
    }


def test_sign():
    assert sign("k", "1", b"{}") == "sha256=" + hmac.new(b"k", b"1.{}", hashlib.sha256).hexdigest()
