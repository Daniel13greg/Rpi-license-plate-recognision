"""``carwash-lpr benchmark``: how many plate reads per second each bay gets on this computer.

Every bay is simulated by a thread that reads plates as fast as it can, all sharing the
one plate reader exactly as the service does. That is the worst case: every bay busy at
the same moment. Without ``--cameras`` the frames are synthetic, so only the plate reader
is measured. With ``--cameras`` the configured cameras are opened and their video
streams decoded, as in production, which on a Pi 5 costs a noticeable share of the CPU.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from carwash_lpr.camera import CameraError, FrameGrabber, FrameSource, create_source
from carwash_lpr.config import CameraConfig, Config
from carwash_lpr.demo import render_scene
from carwash_lpr.recognizer import Recognizer, roi_pixels

ReadOnce = Callable[[], bool]  # performs one read; False if there was no new frame to read

SAMPLE_PLATES = ("KCA 123", "BL AB 123", "ION 7", "OIH 812", "C AB 123", "RMG 001", "MAI 1234", "ABE 456")


def cpu_temperature() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


@dataclass
class Report:
    labels: list[str]  # bay names
    seconds: float  # measured duration of the all-bays-at-once phase
    read_times: list[float]  # durations of single reads with no other bay competing (s)
    reads: list[int]  # reads completed per bay with all bays running
    camera_fps: list[float] | None = None
    temperatures: tuple[float | None, float | None] = (None, None)
    notes: list[str] = field(default_factory=list)

    @property
    def rates(self) -> list[float]:
        return [n / self.seconds for n in self.reads]

    @property
    def total_rate(self) -> float:
        return sum(self.rates)

    @property
    def slowest_rate(self) -> float:
        return min(self.rates) if self.rates else 0.0


def time_reads(read_once: ReadOnce, seconds: float, warmup: int = 2) -> list[float]:
    """Durations of consecutive reads with nothing else competing for the plate reader."""
    for _ in range(warmup):
        read_once()
    durations: list[float] = []
    deadline = time.perf_counter() + seconds
    hard_stop = deadline + 60
    while (time.perf_counter() < deadline or len(durations) < 3) and time.perf_counter() < hard_stop:
        start = time.perf_counter()
        if read_once():
            durations.append(time.perf_counter() - start)
    return durations


def parallel_reads(readers: Sequence[ReadOnce], seconds: float) -> tuple[list[int], float]:
    """Run every bay's reader in its own thread for about ``seconds``.

    Returns the reads completed per bay and the measured duration, which includes the
    reads still running when time was up.
    """
    counts = [0] * len(readers)
    stop = threading.Event()

    def loop(i: int) -> None:
        while not stop.is_set():
            if readers[i]():
                counts[i] += 1

    threads = [threading.Thread(target=loop, args=(i,), daemon=True) for i in range(len(readers))]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    time.sleep(seconds)
    stop.set()
    for thread in threads:
        thread.join(timeout=60)
    return counts, time.monotonic() - started


def run_synthetic(
    recognizer: Recognizer,
    bays: int,
    size: tuple[int, int],
    rois: Sequence[Sequence[float]],
    seconds: float,
) -> Report:
    """Simulated bays reading synthetic frames: the plate reader alone, no video decoding."""
    width, height = size
    frames = [render_scene(SAMPLE_PLATES[i % len(SAMPLE_PLATES)], size=size, seed=i) for i in range(bays)]
    boxes = [roi_pixels(rois[i % len(rois)], width, height) for i in range(bays)]

    def reader(i: int) -> ReadOnce:
        def read_once() -> bool:
            recognizer.recognize(frames[i], boxes[i])
            return True

        return read_once

    readers = [reader(i) for i in range(bays)]
    start_temperature = cpu_temperature()
    read_times = time_reads(readers[0], min(5.0, seconds))
    reads, elapsed = parallel_reads(readers, seconds)
    return Report(
        labels=[f"simulated bay {i + 1}" for i in range(bays)],
        seconds=elapsed,
        read_times=read_times,
        reads=reads,
        temperatures=(start_temperature, cpu_temperature()),
        notes=[
            f"Synthetic {width}x{height} frames: video decoding is not included. Decoding IP camera "
            "streams also uses CPU; measure with --cameras once a camera is installed."
        ],
    )


def run_with_cameras(
    cfg: Config,
    recognizer: Recognizer,
    seconds: float,
    source_factory: Callable[[CameraConfig], FrameSource] = create_source,
    connect_timeout: float = 20.0,
) -> Report:
    """The configured bays with their real cameras: plate reading plus video decoding."""
    grabbers = [FrameGrabber(source_factory(bay.camera), bay.id) for bay in cfg.bays]
    for grabber in grabbers:
        grabber.start()
    try:
        deadline = time.monotonic() + connect_timeout
        while any(g.frames == 0 for g in grabbers) and time.monotonic() < deadline:
            time.sleep(0.1)
        missing = [f"bay {bay.id} ({g.last_error or g.status})" for bay, g in zip(cfg.bays, grabbers) if g.frames == 0]
        if missing:
            raise CameraError(f"no image from {', '.join(missing)}")

        def reader(i: int) -> ReadOnce:
            grabber, roi = grabbers[i], cfg.bays[i].roi
            seen = [0]

            def read_once() -> bool:
                frame, _, seq = grabber.wait_newer(seen[0], timeout=1.0)
                if frame is None or seq == seen[0]:
                    return False
                seen[0] = seq
                recognizer.recognize(frame, roi_pixels(roi, frame.shape[1], frame.shape[0]))
                return True

            return read_once

        start_temperature = cpu_temperature()
        first_frame, _, _ = grabbers[0].latest()
        first_roi = roi_pixels(cfg.bays[0].roi, first_frame.shape[1], first_frame.shape[0])

        def read_same_frame() -> bool:
            recognizer.recognize(first_frame, first_roi)
            return True

        read_times = time_reads(read_same_frame, min(5.0, seconds))
        frames_before = [g.frames for g in grabbers]
        reads, elapsed = parallel_reads([reader(i) for i in range(len(grabbers))], seconds)
        camera_fps = [(g.frames - before) / elapsed for g, before in zip(grabbers, frames_before)]
    finally:
        for grabber in grabbers:
            grabber.stop()
    report = Report(
        labels=[bay.name for bay in cfg.bays],
        seconds=elapsed,
        read_times=read_times,
        reads=reads,
        camera_fps=camera_fps,
        temperatures=(start_temperature, cpu_temperature()),
    )
    if all(rate >= 0.9 * fps for rate, fps in zip(report.rates, camera_fps)):
        report.notes.append(
            "Every bay read every frame its camera sent: the cameras set the pace, the CPU has spare capacity."
        )
    return report


def bays_text(n: int) -> str:
    return f"{n} busy bay" if n == 1 else f"{n} busy bays"


def verdict(rate: float, bays: int, min_reads: int) -> str:
    """Plain-language judgement of the slowest bay's read rate."""
    if rate <= 0:
        return "No plates were read at all: check the cameras and the logs."
    when = f"a car is identified about {min_reads / rate:.1f} s after its plate becomes readable"
    if rate >= 2:
        return f"Good for {bays_text(bays)}: {when}."
    if rate >= 1:
        return (
            f"Workable for {bays_text(bays)}, but tight: {when}. The faster detector "
            "(yolo-v9-t-384), tighter ROIs or lower camera frame rates would add headroom."
        )
    return (
        f"Too slow for {bays_text(bays)}: {when}. Use the yolo-v9-t-384 detector, lower camera "
        "frame rates, fewer bays per Pi, or a second Pi."
    )


def format_report(report: Report, min_reads: int, header: str) -> str:
    times = sorted(report.read_times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))] if times else 0.0
    mean = statistics.mean(times) if times else 0.0
    lines = [
        header,
        "",
        f"One read, bay alone:     {mean * 1000:.0f} ms on average, {p95 * 1000:.0f} ms at worst (95th percentile)",
        f"{'All ' + str(len(report.reads)) + ' bays busy at once' if len(report.reads) > 1 else 'Bay busy non-stop'}: "
        f"{report.total_rate:.1f} reads/s in total",
    ]
    width = max(len(label) for label in report.labels)
    for i, (label, rate) in enumerate(zip(report.labels, report.rates)):
        camera = f"   (camera sent {report.camera_fps[i]:.1f} frames/s)" if report.camera_fps else ""
        lines.append(f"  {label:<{width}}  {rate:5.1f} reads/s{camera}")
    start, end = report.temperatures
    if start is not None and end is not None:
        lines.append(f"CPU temperature: {start:.0f} °C -> {end:.0f} °C (above ~80 °C the Pi slows itself down)")
    lines += ["", verdict(report.slowest_rate, len(report.reads), min_reads)]
    lines += report.notes
    return "\n".join(lines)
