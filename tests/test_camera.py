import time

import cv2
import numpy as np
import pytest

from carwash_lpr.camera import (
    CameraError,
    EndOfStream,
    FrameGrabber,
    FrameSource,
    ImageFolderSource,
    MjpegStreamParser,
    OpenCVSource,
    libcamera_controls,
    rpicam_command,
)
from carwash_lpr.config import CameraConfig
from carwash_lpr.motion import MotionDetector


def jpeg(value: int) -> bytes:
    return cv2.imencode(".jpg", np.full((8, 8, 3), value, np.uint8))[1].tobytes()


def test_mjpeg_parser_handles_arbitrary_chunks():
    a, b = jpeg(10), jpeg(200)
    stream = b"noise" + a + b + a[:10]
    parser = MjpegStreamParser()
    images = []
    for i in range(0, len(stream), 7):
        images += parser.feed(stream[i : i + 7])
    assert images == [a, b]
    assert parser.feed(a[10:]) == [a]


def test_image_folder_source(tmp_path):
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((4, 6, 3), i * 100, np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    source = ImageFolderSource(CameraConfig(type="images", path=str(tmp_path), fps=1000, hflip=True))
    source.open()
    values = [int(source.read()[0, 0, 0]) for _ in range(3)]
    assert values == [0, 100, 0]
    source = ImageFolderSource(CameraConfig(type="images", path=str(tmp_path), fps=1000, loop=False))
    source.open()
    source.read(), source.read()
    with pytest.raises(EndOfStream):
        source.read()


def test_image_folder_source_errors(tmp_path):
    with pytest.raises(CameraError, match="no images"):
        ImageFolderSource(CameraConfig(type="images", path=str(tmp_path))).open()
    with pytest.raises(CameraError, match="does not exist"):
        ImageFolderSource(CameraConfig(type="images", path=str(tmp_path / "missing"))).open()


def test_opencv_source_plays_a_video_file(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("no video encoder available")
    for i in range(3):
        writer.write(np.full((48, 64, 3), i * 80, np.uint8))
    writer.release()
    source = OpenCVSource(CameraConfig(type="opencv", url=path, fps=1000))
    source.open()
    frames = [source.read() for _ in range(5)]  # loops past the end
    assert all(f.shape == (48, 64, 3) for f in frames)
    source.close()


class FlakySource(FrameSource):
    def __init__(self, failures: int, frames: int = 1000):
        self.failures = failures
        self.frames = frames

    def open(self):
        if self.failures:
            self.failures -= 1
            raise CameraError("camera unplugged")

    def read(self):
        if self.frames == 0:
            raise EndOfStream()
        self.frames -= 1
        time.sleep(0.01)
        return np.zeros((4, 4, 3), np.uint8)


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_grabber_reconnects_after_failure():
    grabber = FrameGrabber(FlakySource(failures=1), "test")
    grabber.start()
    try:
        assert wait_for(lambda: grabber.status == "reconnecting")
        assert grabber.last_error == "camera unplugged"
        assert wait_for(lambda: grabber.status == "ok" and grabber.frames > 0)
        frame, _, seq = grabber.wait_newer(grabber.frames, timeout=1.0)
        assert frame is not None and seq > 0
        assert grabber.frame_age() < 1.0
    finally:
        grabber.stop()
    assert not grabber.running


def test_grabber_stops_at_end_of_stream():
    grabber = FrameGrabber(FlakySource(failures=0, frames=3), "test")
    grabber.start()
    assert wait_for(lambda: grabber.status == "ended")
    assert grabber.frames == 3
    grabber.stop()


def test_rpicam_command():
    cfg = CameraConfig(type="rpicam", index=1, width=1280, height=720, fps=8, hflip=True, af_mode="manual",
                       lens_position=0.25, exposure_time_us=2000, extra_args=["--denoise", "cdn_off"])
    cmd = rpicam_command(cfg, "rpicam-vid")
    assert cmd[:7] == ["rpicam-vid", "--timeout", "0", "--nopreview", "--codec", "mjpeg", "--quality"]
    for flag in (["--camera", "1"], ["--width", "1280"], ["--autofocus-mode", "manual"], ["--lens-position", "0.25"],
                 ["--shutter", "2000"], ["--denoise", "cdn_off"]):
        i = cmd.index(flag[0])
        assert cmd[i : i + 2] == flag
    assert "--hflip" in cmd and "--vflip" not in cmd
    assert cmd[cmd.index("--output") + 1] == "-"


def test_libcamera_controls():
    assert libcamera_controls(CameraConfig()) == {}
    assert libcamera_controls(CameraConfig(af_mode="continuous", analogue_gain=2.0)) == {"AfMode": 2, "AnalogueGain": 2.0}
    assert libcamera_controls(CameraConfig(lens_position=0.5)) == {"AfMode": 0, "LensPosition": 0.5}
    assert libcamera_controls(CameraConfig(exposure_time_us=1500, controls={"AeExposureMode": 1})) == {
        "ExposureTime": 1500, "AeExposureMode": 1,
    }


def test_motion_detector():
    motion = MotionDetector()
    still = np.full((480, 640, 3), 90, np.uint8)
    assert motion.update(still) == 1.0  # nothing to compare with yet
    assert motion.update(still) == 0.0
    moved = still.copy()
    moved[100:300, 200:400] = 200
    assert motion.update(moved) > 0.05
    motion.reset()
    assert motion.update(moved) == 1.0
