import os
import subprocess
import sys

import numpy as np
import pytest

from carwash_lpr.config import ConfigError, RecognizerConfig
from carwash_lpr.recognizer import draw_overlay, ensure_models, roi_pixels

from conftest import make_read


def test_roi_pixels():
    assert roi_pixels([0, 0, 1, 1], 1920, 1080) == (0, 0, 1920, 1080)
    assert roi_pixels([0.25, 0.5, 0.75, 1.0], 1920, 1080) == (480, 540, 1440, 1080)
    assert roi_pixels([0.5, 0.5, 0.5001, 0.5001], 100, 100) == (50, 50, 51, 51)  # never empty


def test_draw_overlay_leaves_the_frame_untouched():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_overlay(frame, (10, 10, 600, 400), [make_read("KCA123")], ["KCA 123"])
    assert out.shape == frame.shape
    assert out.any()
    assert not frame.any()


def test_model_problems_are_configuration_errors(tmp_path):
    pytest.importorskip("fast_plate_ocr")
    pytest.importorskip("open_image_models")
    with pytest.raises(ConfigError, match="detector_model: unknown model"):
        ensure_models(RecognizerConfig(detector_model="yolo-v9-t-999-license-plate-end2end"), tmp_path)
    with pytest.raises(ConfigError, match="detector_model: unknown model"):
        ensure_models(RecognizerConfig(detector_model="rf-detr-nano-384-coco"), tmp_path)  # not a plate model
    with pytest.raises(ConfigError, match="not found"):
        ensure_models(RecognizerConfig(detector_model_path=str(tmp_path / "missing.onnx")), tmp_path)
    detector = tmp_path / "detector.onnx"
    detector.write_bytes(b"")
    with pytest.raises(ConfigError, match="ocr_model: unknown model"):
        ensure_models(RecognizerConfig(detector_model_path=str(detector), ocr_model="nope"), tmp_path)


def test_onnx_runtime_telemetry_is_off(tmp_path):
    pytest.importorskip("onnxruntime")
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    env["HOME"] = str(tmp_path)
    code = "import carwash_lpr, onnxruntime, time; time.sleep(1)"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=120)
    assert not (tmp_path / ".cache" / "Microsoft").exists()  # no telemetry device id or queue
