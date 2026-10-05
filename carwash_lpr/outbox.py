"""Durable event queue (SQLite) between the bays and the webhook sender.

Every event is written here first, so nothing is lost when the network or the car wash
server is briefly down. The table doubles as the local history served by the API.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from carwash_lpr.events import PlateEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    bay_id TEXT NOT NULL,
    plate TEXT,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL,
    frame_path TEXT,
    plate_path TEXT,
    status TEXT NOT NULL,  -- pending | delivered | failed | expired | local
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    response_code INTEGER,
    response_body TEXT,
    delivered_at REAL
);
CREATE INDEX IF NOT EXISTS events_pending ON events(status, bay_id, seq);
CREATE INDEX IF NOT EXISTS events_created ON events(created_at);
"""

# The oldest pending event of each bay: events of one bay are delivered strictly in order.
_HEADS = "SELECT MIN(seq) FROM events WHERE status = 'pending' GROUP BY bay_id"


@dataclass
class QueuedEvent:
    seq: int
    event_id: str
    event_type: str
    bay_id: str
    created_at: float
    payload: dict
    frame_path: str | None
    plate_path: str | None
    attempts: int


class Outbox:
    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add(self, event: PlateEvent, payload: dict, deliver: bool) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO events (event_id, event_type, bay_id, plate, created_at, payload, frame_path,"
                " plate_path, status, next_attempt_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id, event.event_type, event.bay_id, event.plate, event.created_at,
                    json.dumps(payload, ensure_ascii=False), event.frame_path, event.plate_path,
                    "pending" if deliver else "local", event.created_at,
                ),
            )

    def due(self, now: float, limit: int = 20) -> list[QueuedEvent]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM events WHERE seq IN ({_HEADS}) AND next_attempt_at <= ? ORDER BY seq LIMIT ?",
                (now, limit),
            ).fetchall()
        return [
            QueuedEvent(
                seq=r["seq"], event_id=r["event_id"], event_type=r["event_type"], bay_id=r["bay_id"],
                created_at=r["created_at"], payload=json.loads(r["payload"]), frame_path=r["frame_path"],
                plate_path=r["plate_path"], attempts=r["attempts"],
            )
            for r in rows
        ]

    def next_attempt_at(self) -> float | None:
        with self._lock:
            row = self._db.execute(f"SELECT MIN(next_attempt_at) FROM events WHERE seq IN ({_HEADS})").fetchone()
        return row[0]

    def _update(self, seq: int, sql: str, *args) -> None:
        with self._lock:
            self._db.execute(f"UPDATE events SET {sql} WHERE seq = ?", (*args, seq))

    def mark_delivered(self, seq: int, now: float, code: int, body: str) -> None:
        self._update(
            seq, "status = 'delivered', attempts = attempts + 1, delivered_at = ?, response_code = ?,"
            " response_body = ?, last_error = NULL", now, code, body,
        )

    def mark_retry(self, seq: int, next_attempt_at: float, error: str, code: int | None = None) -> None:
        self._update(
            seq, "attempts = attempts + 1, next_attempt_at = ?, last_error = ?, response_code = ?",
            next_attempt_at, error, code,
        )

    def mark_failed(self, seq: int, error: str, code: int | None = None, body: str | None = None) -> None:
        self._update(
            seq, "status = 'failed', attempts = attempts + 1, last_error = ?, response_code = ?,"
            " response_body = ?", error, code, body,
        )

    def mark_expired(self, seq: int, error: str) -> None:
        self._update(seq, "status = 'expired', last_error = ?", error)

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()
        return {status: n for status, n in rows}

    def recent(self, limit: int = 50, bay_id: str | None = None) -> list[dict]:
        sql = (
            "SELECT event_id, event_type, bay_id, plate, created_at, status, attempts, last_error,"
            " response_code, response_body, delivered_at, payload, frame_path, plate_path FROM events"
        )
        args: tuple = ()
        if bay_id is not None:
            sql += " WHERE bay_id = ?"
            args = (bay_id,)
        sql += " ORDER BY seq DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, (*args, limit)).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def purge(self, before: float) -> int:
        """Delete finished events older than ``before`` (pending ones are kept)."""
        with self._lock:
            cur = self._db.execute("DELETE FROM events WHERE created_at < ? AND status != 'pending'", (before,))
            return cur.rowcount
