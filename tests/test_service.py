"""The whole service with real threads, HTTP API and webhook, but a fake recognizer."""

import json
import time

import cv2
import numpy as np
import pytest
import requests

from carwash_lpr.service import Service

from conftest import FakeRecognizer, bay_config, make_config, make_read
from test_sender import Receiver

TOKEN = "secret-token"


def wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


@pytest.fixture
def receiver():
    r = Receiver()
    yield r
    r.close()


@pytest.fixture
def service(tmp_path, receiver):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        cv2.imwrite(str(frames / f"{i}.jpg"), np.full((360, 640, 3), 60 + i * 40, np.uint8))
    cfg = make_config(
        tmp_path,
        [bay_config(camera={"type": "images", "path": str(frames), "fps": 20}, process_fps=10)],
        api={"host": "127.0.0.1", "token": TOKEN},
        webhook={"url": receiver.url + "/events", "hmac_secret": "k"},
        device_id="pi-test",
    )
    cfg.api.port = 0  # any free port
    recognizer = FakeRecognizer([make_read("kca 123", box=(100, 100, 360, 160))])
    svc = Service(cfg, recognizer=recognizer)
    svc.start()
    base = f"http://127.0.0.1:{svc.api.port}"
    yield svc, base
    svc.stop()


def auth(**extra):
    return {"headers": {"Authorization": f"Bearer {TOKEN}"}, "timeout": 10, **extra}


def test_plate_is_delivered_to_the_car_wash_system(service, receiver):
    svc, _ = service
    wait_for(lambda: receiver.requests)
    request = receiver.requests[0]
    body = json.loads(request["body"])
    assert body["event_type"] == "plate_recognized"
    assert body["plate"] == "KCA123"
    assert body["plate_display"] == "KCA 123"
    assert body["device_id"] == "pi-test"
    assert request["headers"]["X-LPR-Signature"].startswith("sha256=")
    event = wait_for(lambda: [e for e in svc.outbox.recent(5) if e["status"] == "delivered"])[0]
    assert event["plate"] == "KCA123"
    assert event["frame_path"] and event["plate_path"]  # snapshots were saved


def test_health_needs_no_token_and_shows_no_plates(service):
    _, base = service
    response = requests.get(base + "/health", timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "KCA" not in response.text


def test_api_requires_the_token(service):
    _, base = service
    assert requests.get(base + "/api/v1/status", timeout=10).status_code == 401
    assert requests.get(base + "/api/v1/status", headers={"Authorization": "Bearer nope"}, timeout=10).status_code == 401
    assert requests.get(base + f"/api/v1/status?token={TOKEN}", timeout=10).status_code == 200


def test_status_bays_and_read(service):
    _, base = service
    bay = wait_for(lambda: (b := requests.get(base + "/api/v1/bays/1", **auth()).json())["plate"] and b)
    assert bay["plate_display"] == "KCA 123"
    assert bay["camera"]["status"] == "ok"
    status = requests.get(base + "/api/v1/status", **auth()).json()
    assert status["device_id"] == "pi-test"
    assert status["bays"][0]["id"] == "1"
    read = requests.post(base + "/api/v1/bays/1/read?wait=5", **auth()).json()
    assert read["status"] == "recognized"
    assert read["plate"] == "KCA123"
    assert requests.post(base + "/api/v1/bays/1/read?wait=0", **auth()).status_code == 202
    assert requests.get(base + "/api/v1/bays/9", **auth()).status_code == 404
    assert requests.get(base + "/api/v1/nothing", **auth()).status_code == 404


def test_kept_alive_connection_survives_post_bodies(service):
    _, base = service
    with requests.Session() as session:
        session.headers["Authorization"] = f"Bearer {TOKEN}"
        read = session.post(base + "/api/v1/bays/1/read?wait=0", json={"unused": "body"}, timeout=10)
        assert read.status_code == 202
        status = session.get(base + "/api/v1/status", timeout=10)
        assert status.status_code == 200
        assert status.json()["device_id"] == "pi-test"


def test_snapshot_and_status_page(service):
    _, base = service
    response = wait_for(lambda: (r := requests.get(base + "/api/v1/bays/1/snapshot.jpg", **auth())).ok and r)
    assert response.headers["Content-Type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (360, 640, 3)
    page = requests.get(base + "/", **auth())
    assert "text/html" in page.headers["Content-Type"]


def test_simulated_event(service, receiver):
    svc, base = service
    response = requests.post(base + "/api/v1/bays/1/simulate", json={"plate": "c ab 123"}, **auth())
    assert response.status_code == 202
    assert response.json()["plate"] == "CAB123"
    wait_for(lambda: any(json.loads(r["body"])["plate"] == "CAB123" for r in receiver.requests))
    events = requests.get(base + "/api/v1/events?limit=10&bay=1", **auth()).json()
    simulated = [e for e in events if e["plate"] == "CAB123"][0]
    assert simulated["payload"]["trigger"] == "simulated"
    bad = requests.post(base + "/api/v1/bays/1/simulate", json={"event_type": "boom"}, **auth())
    assert bad.status_code == 400


def test_problems_and_housekeeping(service):
    svc, _ = service
    wait_for(lambda: svc.bays["1"].stats["recognitions"] > 0)
    assert svc.problems() == []
    svc.housekeeping()
