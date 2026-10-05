from carwash_lpr.events import PlateEvent
from carwash_lpr.outbox import Outbox


def add(outbox, bay, plate, created, deliver=True):
    event = PlateEvent("plate_recognized", bay, plate=plate, created_at=created)
    outbox.add(event, event.payload("pi"), deliver)
    return event


def test_due_returns_oldest_event_per_bay(tmp_path):
    outbox = Outbox(tmp_path / "events.db")
    a1 = add(outbox, "1", "AAA111", 100.0)
    add(outbox, "1", "AAA222", 101.0)
    b1 = add(outbox, "2", "BBB111", 102.0)
    add(outbox, "1", "LOCAL1", 103.0, deliver=False)
    due = outbox.due(200.0)
    assert [e.event_id for e in due] == [a1.event_id, b1.event_id]
    assert due[0].payload["plate"] == "AAA111"


def test_retry_holds_back_only_that_bay(tmp_path):
    outbox = Outbox(tmp_path / "events.db")
    a1 = add(outbox, "1", "AAA111", 100.0)
    add(outbox, "1", "AAA222", 101.0)
    b1 = add(outbox, "2", "BBB111", 102.0)
    first = outbox.due(200.0)[0]
    outbox.mark_retry(first.seq, 260.0, "timeout")
    assert [e.event_id for e in outbox.due(200.0)] == [b1.event_id]
    assert outbox.next_attempt_at() == 102.0
    later = outbox.due(261.0)
    assert later[0].event_id == a1.event_id
    assert later[0].attempts == 1


def test_delivery_moves_the_queue_on(tmp_path):
    outbox = Outbox(tmp_path / "events.db")
    add(outbox, "1", "AAA111", 100.0)
    a2 = add(outbox, "1", "AAA222", 101.0)
    outbox.mark_delivered(outbox.due(200.0)[0].seq, 200.0, 200, "ok")
    assert [e.event_id for e in outbox.due(200.0)] == [a2.event_id]
    outbox.mark_failed(outbox.due(200.0)[0].seq, "HTTP 400", 400, "bad")
    assert outbox.due(200.0) == []
    assert outbox.next_attempt_at() is None
    assert outbox.counts() == {"delivered": 1, "failed": 1}


def test_recent_and_purge(tmp_path):
    outbox = Outbox(tmp_path / "events.db")
    add(outbox, "1", "AAA111", 100.0, deliver=False)
    add(outbox, "2", "BBB111", 200.0, deliver=False)
    pending = add(outbox, "2", "BBB222", 50.0)
    assert [e["plate"] for e in outbox.recent(10)] == ["BBB222", "BBB111", "AAA111"]
    assert [e["plate"] for e in outbox.recent(10, "1")] == ["AAA111"]
    assert outbox.recent(1)[0]["payload"]["plate"] == "BBB222"
    assert outbox.purge(150.0) == 1  # AAA111; the old pending event is kept
    assert {e["event_id"] for e in outbox.recent(10)} >= {pending.event_id}


def test_survives_reopening(tmp_path):
    outbox = Outbox(tmp_path / "events.db")
    event = add(outbox, "1", "AAA111", 100.0)
    outbox.close()
    reopened = Outbox(tmp_path / "events.db")
    assert reopened.due(200.0)[0].event_id == event.event_id
