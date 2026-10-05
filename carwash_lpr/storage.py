"""JPEG snapshots of every event (evidence for disputes, material for tuning the camera).

Snapshots show vehicles and people, i.e. personal data: keep the retention short and
tell customers about the cameras (Moldovan Law 195/2024 on personal data protection,
in force since 23 August 2026).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2

from carwash_lpr.config import StorageConfig
from carwash_lpr.events import PlateEvent

log = logging.getLogger(__name__)

MAX_FRAME_WIDTH = 1280


class SnapshotStore:
    def __init__(self, root: str | Path, cfg: StorageConfig):
        self.root = Path(root)
        self.cfg = cfg

    def save(self, event: PlateEvent) -> None:
        """Write the event's frame and plate crop; sets event.frame_path / plate_path."""
        if event.frame is None and event.crop is None:
            return
        moment = datetime.fromtimestamp(event.created_at)
        folder = self.root / moment.strftime("%Y-%m-%d") / event.bay_id
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{moment:%H%M%S}_{event.event_type}_{event.plate or 'none'}_{event.event_id[:8]}"
        if event.frame is not None:
            frame = event.frame
            if frame.shape[1] > MAX_FRAME_WIDTH:
                height = round(frame.shape[0] * MAX_FRAME_WIDTH / frame.shape[1])
                frame = cv2.resize(frame, (MAX_FRAME_WIDTH, height), interpolation=cv2.INTER_AREA)
            path = folder / f"{stem}.jpg"
            if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                event.frame_path = str(path)
        if event.crop is not None and event.crop.size:
            path = folder / f"{stem}_plate.jpg"
            if cv2.imwrite(str(path), event.crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                event.plate_path = str(path)

    def cleanup(self, now: float | None = None) -> int:
        """Delete snapshots past retention, then the oldest ones while over the size cap."""
        if not self.root.is_dir():
            return 0
        now = time.time() if now is None else now
        cutoff = now - self.cfg.snapshot_retention_days * 86_400
        limit = self.cfg.max_snapshot_mb * 1024 * 1024
        files = []
        for path in self.root.rglob("*.jpg"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
        files.sort()
        total = sum(size for _, size, _ in files)
        removed = 0
        for mtime, size, path in files:
            if mtime >= cutoff and total <= limit:
                break
            try:
                path.unlink()
                removed += 1
                total -= size
            except OSError as exc:
                log.warning("cannot delete %s: %s", path, exc)
        for folder in sorted((p for p in self.root.rglob("*") if p.is_dir()), reverse=True):
            try:
                folder.rmdir()  # only succeeds for empty folders
            except OSError:
                pass
        if removed:
            log.info("deleted %d old snapshots", removed)
        return removed
