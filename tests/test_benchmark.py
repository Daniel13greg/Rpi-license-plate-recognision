import threading
import time

import cv2
import numpy as np
import pytest

from carwash_lpr import benchmark
from carwash_lpr.camera import CameraError
from carwash_lpr.cli import main
from carwash_lpr.recognizer import FairLock

from conftest import bay_config, make_config


class SlowRecognizer:
    """Takes ``delay`` seconds per read and, like the real one, serves one bay at a time."""

    def __init__(self, delay: float = 0.005):
        self.delay = delay
        self.lock = FairLock()

    def recognize(self, frame, roi=None):
        with self.lock:
            time.sleep(self.delay)
        return []


def test_fair_lock_takes_turns_and_excludes():
    lock = FairLock()
    counts = [0] * 4
    inside = []
    start = threading.Barrier(4)
    stop = threading.Event()

    def work(i):
        start.wait()
        while not stop.is_set():
            with lock:
                inside.append(i)
                assert len(inside) == 1
                counts[i] += 1
                time.sleep(0.001)
                inside.pop()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join()
    assert min(counts) > 10
    assert max(counts) - min(counts) <= 3


def test_simulated_bays_share_the_reader_equally():
    report = benchmark.run_synthetic(SlowRecognizer(0.005), 3, (640, 360), [[0, 0, 1, 1]], seconds=0.6)
    assert len(report.reads) == 3
    assert min(report.reads) > 0
    assert max(report.reads) - min(report.reads) <= 3
    assert 30 < report.total_rate < 250  # at most ~200 reads/s at 5 ms each
    assert report.read_times and all(t >= 0.005 for t in report.read_times)
    text = benchmark.format_report(report, 2, "Plate reader: test")
    assert "simulated bay 3" in text
    assert "reads/s in total" in text
    assert "video decoding is not included" in text


def test_benchmark_with_the_configured_cameras(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        cv2.imwrite(str(frames / f"{i}.jpg"), np.full((360, 640, 3), 50 * i, np.uint8))
    camera = {"type": "images", "path": str(frames), "fps": 20}
    cfg = make_config(tmp_path, [bay_config(id="1", name="Box 1", camera=camera), bay_config(id="2", name="Box 2", camera=camera)])
    report = benchmark.run_with_cameras(cfg, SlowRecognizer(0.001), seconds=1.0)
    assert report.labels == ["Box 1", "Box 2"]
    assert all(10 < fps < 30 for fps in report.camera_fps)
    # a bay cannot read more frames than its camera sends
    assert all(rate <= fps + 2 for rate, fps in zip(report.rates, report.camera_fps))
    assert any("cameras set the pace" in note for note in report.notes)
    assert "camera sent" in benchmark.format_report(report, 2, "header")


def test_benchmark_names_a_camera_without_images(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = make_config(tmp_path, [bay_config(camera={"type": "images", "path": str(empty)})])
    with pytest.raises(CameraError, match="bay 1"):
        benchmark.run_with_cameras(cfg, SlowRecognizer(), seconds=0.5, connect_timeout=0.5)


@pytest.mark.parametrize("rate, start", [(3.0, "Good"), (1.5, "Workable"), (0.5, "Too slow"), (0.0, "No plates")])
def test_verdict(rate, start):
    assert benchmark.verdict(rate, 5, 2).startswith(start)


def test_verdict_estimates_identification_time():
    assert "about 1.0 s" in benchmark.verdict(2.0, 5, 2)


def test_cameras_option_needs_a_config(capsys):
    assert main(["benchmark", "--cameras"]) == 2
    assert "--cameras needs" in capsys.readouterr().err
