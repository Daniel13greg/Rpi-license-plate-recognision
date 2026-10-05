"""Camera sources and a background grabber that always holds the newest frame.

Sources:
  picamera2  Raspberry Pi camera modules through the Picamera2 library (python3-picamera2)
  rpicam     Raspberry Pi camera modules through an ``rpicam-vid`` MJPEG subprocess
             (no Python camera bindings needed)
  opencv     USB webcams (/dev/videoN), IP cameras (rtsp://, http://) and video files
  images     a folder of pictures, for testing without a camera

All sources return BGR ``numpy`` frames, the layout OpenCV and the models expect.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from carwash_lpr.config import CameraConfig

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CameraError(RuntimeError):
    pass


class EndOfStream(CameraError):
    """A video file or image folder without ``loop`` has no more frames."""


class FrameSource:
    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> np.ndarray:
        """Block until the next frame; raise CameraError if the camera is gone."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return type(self).__name__


def _flip(frame: np.ndarray, cfg: CameraConfig) -> np.ndarray:
    if cfg.hflip and cfg.vflip:
        return cv2.flip(frame, -1)
    if cfg.hflip:
        return cv2.flip(frame, 1)
    if cfg.vflip:
        return cv2.flip(frame, 0)
    return frame


class _Pacer:
    """Sleeps so that file based sources play back at the configured frame rate."""

    def __init__(self, fps: float):
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self.next_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if self.next_at > now:
            time.sleep(self.next_at - now)
            now = self.next_at
        self.next_at = now + self.interval


class OpenCVSource(FrameSource):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None
        self.is_file = False
        self.pacer = _Pacer(cfg.fps)

    def describe(self) -> str:
        return f"opencv {self.cfg.url or f'/dev/video{self.cfg.index}'}"

    def open(self) -> None:
        cfg = self.cfg
        target: str | int = cfg.url if cfg.url else cfg.index
        device = isinstance(target, int) or bool(re.fullmatch(r"/dev/video\d+", str(target)))
        self.is_file = isinstance(target, str) and os.path.isfile(target)
        if device:
            cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
            # USB webcams only deliver high resolutions at a usable rate as MJPEG.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if str(target).startswith("rtsp"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{cfg.rtsp_transport}"
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000]
            try:
                cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG, params)
            except (cv2.error, TypeError):
                cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"cannot open {target}")
        self.cap = cap

    def read(self) -> np.ndarray:
        assert self.cap is not None
        if self.is_file:
            self.pacer.wait()
        ok, frame = self.cap.read()
        if not ok and self.is_file:
            if not self.cfg.loop:
                raise EndOfStream("end of video file")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("no frame from camera")
        return _flip(frame, self.cfg)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ImageFolderSource(FrameSource):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.files: list[Path] = []
        self.position = 0
        self.pacer = _Pacer(cfg.fps)

    def describe(self) -> str:
        return f"images {self.cfg.path}"

    def open(self) -> None:
        folder = Path(self.cfg.path or "")
        if not folder.is_dir():
            raise CameraError(f"image folder {folder} does not exist")
        self.files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.files:
            raise CameraError(f"no images in {folder}")
        self.position = 0

    def read(self) -> np.ndarray:
        self.pacer.wait()
        if self.position >= len(self.files):
            if not self.cfg.loop:
                raise EndOfStream("no more images")
            self.position = 0
        path = self.files[self.position]
        self.position += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise CameraError(f"cannot read image {path}")
        return _flip(frame, self.cfg)


def libcamera_controls(cfg: CameraConfig) -> dict:
    """libcamera control values for the configured focus/exposure settings."""
    controls: dict = {}
    if cfg.af_mode == "manual":
        controls["AfMode"] = 0
        if cfg.lens_position is not None:
            controls["LensPosition"] = cfg.lens_position
    elif cfg.af_mode == "auto":
        controls["AfMode"] = 1
    elif cfg.af_mode == "continuous":
        controls["AfMode"] = 2
    elif cfg.lens_position is not None:
        controls.update(AfMode=0, LensPosition=cfg.lens_position)
    if cfg.exposure_time_us is not None:
        controls["ExposureTime"] = cfg.exposure_time_us
    if cfg.analogue_gain is not None:
        controls["AnalogueGain"] = cfg.analogue_gain
    controls.update(cfg.controls)
    return controls


class Picamera2Source(FrameSource):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cam = None

    def describe(self) -> str:
        return f"picamera2 camera {self.cfg.index}"

    def open(self) -> None:
        try:
            from libcamera import Transform
            from picamera2 import Picamera2
        except ImportError as exc:
            raise CameraError(
                "picamera2 is not available: install python3-picamera2 and create the virtualenv "
                "with --system-site-packages, or use camera type 'rpicam'"
            ) from exc
        cfg = self.cfg
        cam = Picamera2(cfg.index)
        try:
            config = cam.create_video_configuration(
                # "RGB888" frames are stored B, G, R: exactly what OpenCV expects.
                main={"size": (cfg.width, cfg.height), "format": "RGB888"},
                transform=Transform(hflip=int(cfg.hflip), vflip=int(cfg.vflip)),
                controls={"FrameRate": cfg.fps},
                buffer_count=4,
            )
            cam.configure(config)
            cam.start()
            controls = libcamera_controls(cfg)
            supported = getattr(cam, "camera_controls", None) or {}
            unsupported = [name for name in controls if supported and name not in supported]
            for name in unsupported:
                log.warning("camera %d does not support control %s; ignoring it", cfg.index, name)
                controls.pop(name)
            if controls:
                cam.set_controls(controls)
        except Exception:
            cam.close()
            raise
        self.cam = cam

    def read(self) -> np.ndarray:
        frame = self.cam.capture_array("main")
        if frame is None:
            raise CameraError("no frame from camera")
        return frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] == 4 else frame

    def close(self) -> None:
        if self.cam is not None:
            try:
                self.cam.stop()
            finally:
                self.cam.close()
                self.cam = None


def rpicam_command(cfg: CameraConfig, executable: str) -> list[str]:
    cmd = [
        executable, "--timeout", "0", "--nopreview", "--codec", "mjpeg", "--quality", "90",
        "--camera", str(cfg.index), "--width", str(cfg.width), "--height", str(cfg.height),
        "--framerate", str(cfg.fps), "--flush", "--output", "-",
    ]
    if cfg.hflip:
        cmd.append("--hflip")
    if cfg.vflip:
        cmd.append("--vflip")
    if cfg.af_mode != "default":
        cmd += ["--autofocus-mode", cfg.af_mode]
    if cfg.lens_position is not None:
        cmd += ["--lens-position", str(cfg.lens_position)]
    if cfg.exposure_time_us is not None:
        cmd += ["--shutter", str(cfg.exposure_time_us)]
    if cfg.analogue_gain is not None:
        cmd += ["--gain", str(cfg.analogue_gain)]
    return cmd + list(cfg.extra_args)


class MjpegStreamParser:
    """Splits a concatenated MJPEG byte stream into individual JPEG images."""

    MAX_BUFFER = 32 * 1024 * 1024

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer += data
        images = []
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                # keep a trailing 0xFF: it may be the first half of the next start marker
                del self.buffer[: max(0, len(self.buffer) - 1)]
                break
            end = self.buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                del self.buffer[:start]
                if len(self.buffer) > self.MAX_BUFFER:
                    self.buffer.clear()
                break
            images.append(bytes(self.buffer[start : end + 2]))
            del self.buffer[: end + 2]
        return images


class RpicamSource(FrameSource):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.parser = MjpegStreamParser()
        self.pending: deque[bytes] = deque()
        self.stderr_tail: deque[str] = deque(maxlen=20)

    def describe(self) -> str:
        return f"rpicam-vid camera {self.cfg.index}"

    def open(self) -> None:
        executable = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if executable is None:
            raise CameraError("rpicam-vid not found (sudo apt install rpicam-apps)")
        cmd = rpicam_command(self.cfg, executable)
        log.debug("starting %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.parser = MjpegStreamParser()
        self.pending.clear()
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for line in iter(proc.stderr.readline, b""):
            text = line.decode(errors="replace").rstrip()
            self.stderr_tail.append(text)
            log.debug("rpicam-vid: %s", text)

    def read(self) -> np.ndarray:
        assert self.proc is not None
        while True:
            if self.pending:
                jpeg = self.pending.pop()  # only the newest frame matters
                self.pending.clear()
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                continue
            data = self.proc.stdout.read(256 * 1024)
            if not data:
                time.sleep(0.2)  # let stderr reach the tail buffer
                detail = " | ".join(list(self.stderr_tail)[-3:])
                raise CameraError(f"rpicam-vid stopped: {detail or 'no output'}")
            self.pending.extend(self.parser.feed(data))

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def create_source(cfg: CameraConfig) -> FrameSource:
    sources = {
        "picamera2": Picamera2Source,
        "rpicam": RpicamSource,
        "opencv": OpenCVSource,
        "images": ImageFolderSource,
    }
    return sources[cfg.type](cfg)


class FrameGrabber:
    """Reads a source on its own thread so the newest frame is always ready.

    Lost cameras are reopened with exponential backoff. Processing never waits for
    old frames to drain, which matters for IP cameras that buffer.
    """

    def __init__(self, source: FrameSource, name: str):
        self.source = source
        self.name = name
        self.status = "starting"
        self.last_error = ""
        self.frames = 0
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0  # time.monotonic() of the newest frame
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._cond:
            return self._frame, self._frame_time, self.frames

    def frame_age(self) -> float | None:
        with self._cond:
            return None if self._frame is None else time.monotonic() - self._frame_time

    def wait_newer(self, seen: int, timeout: float) -> tuple[np.ndarray | None, float, int]:
        """Wait until a frame newer than sequence number ``seen`` arrives (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self.frames > seen or self._stop.is_set(), timeout)
            return self._frame, self._frame_time, self.frames

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.source.open()
                self.status = "ok"
                log.info("camera %s connected (%s)", self.name, self.source.describe())
                backoff = 1.0
                while not self._stop.is_set():
                    frame = self.source.read()
                    with self._cond:
                        self._frame = frame
                        self._frame_time = time.monotonic()
                        self.frames += 1
                        self._cond.notify_all()
            except EndOfStream:
                self.status = "ended"
                log.info("camera %s: end of input", self.name)
                return
            except Exception as exc:  # cameras fail in many library specific ways
                self.status = "reconnecting"
                self.last_error = str(exc)
                log.error("camera %s: %s (retrying in %.0fs)", self.name, exc, backoff)
            finally:
                try:
                    self.source.close()
                except Exception as exc:
                    log.debug("camera %s: error while closing: %s", self.name, exc)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        self.status = "stopped"
