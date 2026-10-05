import os

import numpy as np

from carwash_lpr.config import StorageConfig
from carwash_lpr.events import PlateEvent
from carwash_lpr.storage import SnapshotStore


def test_save_writes_frame_and_plate(tmp_path):
    store = SnapshotStore(tmp_path, StorageConfig())
    event = PlateEvent(
        "plate_recognized", "1", plate="KCA123",
        frame=np.zeros((1080, 1920, 3), np.uint8), crop=np.zeros((60, 260, 3), np.uint8),
    )
    store.save(event)
    assert os.path.isfile(event.frame_path)
    assert os.path.isfile(event.plate_path)
    assert "KCA123" in event.frame_path
    import cv2

    assert cv2.imread(event.frame_path).shape[1] == 1280  # downscaled


def test_save_without_images_does_nothing(tmp_path):
    store = SnapshotStore(tmp_path, StorageConfig())
    event = PlateEvent("vehicle_left", "1")
    store.save(event)
    assert event.frame_path is None
    assert list(tmp_path.iterdir()) == []


def test_cleanup_by_age_and_size(tmp_path):
    store = SnapshotStore(tmp_path, StorageConfig(snapshot_retention_days=1, max_snapshot_mb=10))
    folder = tmp_path / "2026-01-01" / "1"
    folder.mkdir(parents=True)
    now = 10 * 86_400.0
    old = folder / "old.jpg"
    old.write_bytes(b"x" * 100)
    os.utime(old, (now - 2 * 86_400, now - 2 * 86_400))
    fresh = folder / "fresh.jpg"
    fresh.write_bytes(b"x" * 100)
    os.utime(fresh, (now - 3600, now - 3600))
    assert store.cleanup(now) == 1
    assert not old.exists() and fresh.exists()

    big = folder / "big.jpg"
    big.write_bytes(b"x" * (10 * 1024 * 1024 - 50))  # with fresh.jpg just over the 10 MB cap
    os.utime(big, (now, now))
    assert store.cleanup(now) == 1  # the oldest file goes first
    assert not fresh.exists() and big.exists()


def test_cleanup_removes_empty_folders(tmp_path):
    store = SnapshotStore(tmp_path, StorageConfig(snapshot_retention_days=1))
    folder = tmp_path / "2020-01-01" / "1"
    folder.mkdir(parents=True)
    f = folder / "a.jpg"
    f.write_bytes(b"x")
    os.utime(f, (0, 0))
    store.cleanup(10 * 86_400.0)
    assert not (tmp_path / "2020-01-01").exists()
