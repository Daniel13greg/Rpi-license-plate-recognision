"""End to end with the real detector and OCR models on synthetic Moldovan plates.

Needs the models (downloaded on first use, ~11 MB); skipped when they are unavailable.
"""

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("fast_plate_ocr")
pytest.importorskip("open_image_models")

from carwash_lpr import plates  # noqa: E402
from carwash_lpr.config import RecognizerConfig  # noqa: E402
from carwash_lpr.demo import render_scene, write_demo_images  # noqa: E402
from carwash_lpr.recognizer import PlateRecognizer  # noqa: E402
from carwash_lpr.service import Service  # noqa: E402

from conftest import bay_config, make_config  # noqa: E402
from test_sender import Receiver  # noqa: E402


@pytest.fixture(scope="module")
def recognizer():
    models = Path(os.environ.get("CARWASH_LPR_MODELS", Path.home() / ".cache" / "carwash-lpr" / "models"))
    try:
        return PlateRecognizer(RecognizerConfig(threads=2), models)
    except Exception as exc:  # offline, no ONNX runtime for this platform, ...
        pytest.skip(f"recognition models unavailable: {exc}")


def plates_in(recognizer, frame):
    found = []
    for read in recognizer.recognize(frame):
        interpretation = plates.interpret(read.text)
        if interpretation is not None:
            found.append(interpretation.plate)
    return found


@pytest.mark.parametrize(
    "text, plate",
    [("KCA 123", "KCA123"), ("BL AB 123", "BLAB123"), ("ION 7", "ION7"), ("MAI 1234", "MAI1234"), ("RMG 001", "RMG001")],
)
def test_reads_synthetic_moldovan_plates(recognizer, text, plate):
    assert plate in plates_in(recognizer, render_scene(text))


def test_reads_green_ev_plate(recognizer):
    assert "EVA777" in plates_in(recognizer, render_scene("EVA 777", green=True))


def test_roi_excludes_plates_outside(recognizer):
    frame = render_scene("KCA 123")
    assert recognizer.recognize(frame, roi=(0, 0, 400, 300)) == []


def test_empty_bay_has_no_plates(recognizer):
    assert plates_in(recognizer, render_scene(None)) == []


def test_whole_pipeline_from_camera_to_webhook(tmp_path, recognizer):
    write_demo_images(tmp_path / "frames", ["KCA 123", "BL AB 123"], frames_per_car=16, empty_frames=8)
    receiver = Receiver()
    cfg = make_config(
        tmp_path,
        [bay_config(camera={"type": "images", "path": str(tmp_path / "frames"), "fps": 8, "loop": False},
                    motion={"enabled": True}, process_fps=8)],
        api={"enabled": False},
        webhook={"url": receiver.url},
    )
    service = Service(cfg, recognizer=recognizer)
    service.start()
    try:
        deadline = time.monotonic() + 60
        while len(receiver.requests) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        service.stop()
        receiver.close()
    events = [json.loads(r["body"]) for r in receiver.requests]
    assert [(e["event_type"], e["plate"]) for e in events] == [
        ("plate_recognized", "KCA123"),
        ("plate_recognized", "BLAB123"),
    ]
    assert events[1]["plate_display"] == "BL AB 123"
    assert events[0]["confidence"] > 0.8
