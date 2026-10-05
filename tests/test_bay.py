import numpy as np
import pytest

from carwash_lpr.bay import BayWorker

from conftest import FakeGrabber, FakeRecognizer, FakeSensor, bay_config, make_config, make_read

FPS = 4


@pytest.fixture
def setup(tmp_path, clock):
    def build(recognizer, sensor=None, **bay):
        cfg = make_config(tmp_path, [bay_config(**bay)])
        events = []
        grabber = FakeGrabber()
        worker = BayWorker(cfg.bays[0], cfg, recognizer, grabber, events.append, sensor, clock)

        def advance(seconds: float, new_frames: bool = True):
            for _ in range(round(seconds * FPS)):
                clock.now += 1 / FPS
                if new_frames:
                    grabber.next_frame()
                worker.step(clock.now)

        return worker, events, advance

    return build


def types(events):
    return [e.event_type for e in events]


# --- continuous mode ---------------------------------------------------------------------


def test_arrival_is_reported_once(setup):
    recognizer = FakeRecognizer([make_read("KCA 123")])
    worker, events, advance = setup(recognizer)
    advance(10)
    assert types(events) == ["plate_recognized"]
    event = events[0]
    assert (event.plate, event.plate_display, event.plate_format, event.country) == ("KCA123", "KCA 123", "md_standard", "MD")
    assert event.votes == 2
    assert event.trigger == "continuous"
    assert event.frame is not None and event.crop is not None
    assert event.candidates[0]["plate"] == "KCA123"
    status = worker.status()
    assert status["state"] == "occupied"
    assert status["plate"] == "KCA123"
    assert status["last_event"]["plate"] == "KCA123"


def test_single_misread_is_outvoted(setup):
    sequence = iter(["KCA123", "KCA128", "KCA123", "KCA123"] * 10)
    recognizer = FakeRecognizer(lambda frame: [make_read(next(sequence))])
    _, events, advance = setup(recognizer)
    advance(5)
    assert [e.plate for e in events] == ["KCA123"]


def test_reads_that_are_not_plates_are_ignored(setup):
    recognizer = FakeRecognizer([make_read("STOP"), make_read("KCA123", confidence=0.3)])
    _, events, advance = setup(recognizer)
    advance(5)
    assert events == []


def test_departure_and_repeat_cooldown(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(
        recognizer,
        presence={"absence_timeout_seconds": 10, "repeat_cooldown_seconds": 60, "report_departures": True},
    )
    advance(2)
    recognizer.reads = []
    advance(11)
    assert types(events) == ["plate_recognized", "vehicle_left"]
    assert events[1].plate == "KCA123"
    assert worker.status()["state"] == "empty"

    # back within the cooldown (e.g. the plate was hidden by foam): nothing new is reported
    recognizer.reads = [make_read("KCA123")]
    advance(2)
    recognizer.reads = []
    advance(11)
    assert types(events) == ["plate_recognized", "vehicle_left"]

    # after the cooldown it is a new visit
    advance(60)
    recognizer.reads = [make_read("KCA123")]
    advance(2)
    assert types(events) == ["plate_recognized", "vehicle_left", "plate_recognized"]


def test_next_car_replaces_the_previous_one(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    _, events, advance = setup(recognizer, presence={"report_departures": True})
    advance(2)
    recognizer.reads = [make_read("BLAB123")]
    advance(4)
    assert types(events) == ["plate_recognized", "vehicle_left", "plate_recognized"]
    assert [e.plate for e in events] == ["KCA123", "KCA123", "BLAB123"]
    assert events[2].plate_display == "BL AB 123"


def test_departures_are_not_reported_by_default(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    _, events, advance = setup(recognizer, presence={"absence_timeout_seconds": 5})
    advance(2)
    recognizer.reads = []
    advance(10)
    assert types(events) == ["plate_recognized"]


def test_motion_gating_skips_static_frames(setup):
    recognizer = FakeRecognizer([])
    worker, _, advance = setup(recognizer, motion={"enabled": True, "hold_seconds": 0, "idle_interval_seconds": 0})
    grabber = worker.grabber
    grabber.frame = np.full((720, 1280, 3), 100, np.uint8)
    advance(5)
    assert recognizer.calls == 1  # only the very first frame counts as motion
    changed = grabber.frame.copy()
    changed[200:600, 300:1000] = 220
    grabber.frame = changed
    advance(0.25)
    assert recognizer.calls == 2


def test_idle_interval_still_checks_now_and_then(setup):
    recognizer = FakeRecognizer([])
    worker, _, advance = setup(recognizer, motion={"enabled": True, "hold_seconds": 0, "idle_interval_seconds": 2})
    worker.grabber.frame = np.full((720, 1280, 3), 100, np.uint8)
    advance(10)
    assert 5 <= recognizer.calls <= 7


def test_read_request_continuous(setup):
    recognizer = FakeRecognizer([])
    worker, events, advance = setup(recognizer)
    request = worker.request_read()
    advance(1)
    assert not request.done
    recognizer.reads = [make_read("KCA123")]
    advance(1)
    assert request.done
    assert request.result["plate"] == "KCA123"
    assert request.result["event_id"] == events[0].event_id
    # a car that is already known is answered straight away
    second = worker.request_read()
    advance(0.25)
    assert second.result["plate"] == "KCA123"
    assert second.result["status"] == "recognized"


def test_known_car_is_only_rechecked(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(recognizer)  # motion gating off: normally every frame is read
    advance(1)
    assert len(events) == 1
    calls = recognizer.calls
    advance(10)
    assert recognizer.calls - calls == 5  # one check every 2 s instead of 40 reads
    assert worker.status()["recognition"] == "recheck"
    assert worker.status()["state"] == "occupied"


def test_another_plate_brings_back_full_speed(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(recognizer)
    advance(3)
    recognizer.reads = [make_read("BLAB123")]
    advance(2.5)  # the next recheck sees the new plate, then every frame is read again
    assert [e.plate for e in events] == ["KCA123", "BLAB123"]


def test_waiting_api_caller_gets_full_speed(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, _, advance = setup(recognizer)
    advance(1)
    recognizer.reads = []  # foam hides the plate for longer than FRESH_SECONDS
    advance(12)
    calls = recognizer.calls
    request = worker.request_read()
    advance(1)
    assert recognizer.calls - calls == 4  # every frame while the caller waits
    recognizer.reads = [make_read("KCA123")]
    advance(1)
    assert request.result["plate"] == "KCA123"


def test_rechecking_can_be_turned_off(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    _, _, advance = setup(recognizer, presence={"recheck_interval_seconds": 0})
    advance(5)
    assert recognizer.calls == 20


def test_no_new_frames_means_no_recognition(setup):
    recognizer = FakeRecognizer([make_read("KCA123")])
    _, events, advance = setup(recognizer)
    advance(0.25)
    advance(5, new_frames=False)
    assert recognizer.calls == 1
    assert events == []


# --- trigger mode --------------------------------------------------------------------------

TRIGGER = {
    "gpio_pin": 17,
    "activate_delay_seconds": 0.5,
    "release_delay_seconds": 2,
    "window_seconds": 5,
    "pre_trigger_seconds": 0,
}


def test_sensor_session(setup):
    sensor = FakeSensor()
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(recognizer, sensor, mode="trigger", trigger=TRIGGER)
    advance(2)
    assert recognizer.calls == 0  # nothing to do without a car
    sensor.active = True
    advance(0.25)
    assert worker.status()["state"] == "idle"  # still debouncing
    advance(1)
    assert types(events) == ["plate_recognized"]
    assert events[0].trigger == "gpio"
    calls = recognizer.calls
    advance(5)
    assert recognizer.calls == calls  # car already read: the CPU rests
    assert worker.status()["state"] == "done"

    sensor.active = False
    advance(1)
    assert len(events) == 1  # release delay not over yet
    advance(2)
    assert types(events) == ["plate_recognized", "vehicle_left"]
    assert events[1].plate == "KCA123"
    assert events[1].duration_seconds > 0
    assert worker.status()["state"] == "idle"


def test_sensor_blip_is_ignored(setup):
    sensor = FakeSensor()
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(recognizer, sensor, mode="trigger", trigger=TRIGGER)
    sensor.active = True
    advance(0.25)
    sensor.active = False
    advance(3)
    assert events == []
    assert recognizer.calls == 0


def test_unreadable_plate_is_reported(setup):
    sensor = FakeSensor()
    recognizer = FakeRecognizer([make_read("KCA123", confidence=0.65)])  # never confident enough
    worker, events, advance = setup(recognizer, sensor, mode="trigger", trigger=TRIGGER)
    sensor.active = True
    advance(7)
    assert types(events) == ["plate_unrecognized"]
    assert events[0].plate is None
    assert events[0].candidates[0]["plate"] == "KCA123"
    assert events[0].frame is not None
    sensor.active = False
    advance(3)
    assert types(events) == ["plate_unrecognized", "vehicle_left"]
    assert events[1].plate is None


def test_http_trigger_without_sensor(setup):
    recognizer = FakeRecognizer([make_read("C AB 123")])
    worker, events, advance = setup(recognizer, mode="trigger", trigger={**TRIGGER, "gpio_pin": None})
    advance(1)
    assert recognizer.calls == 0
    request = worker.request_read("http")
    advance(1)
    assert request.result["status"] == "recognized"
    assert request.result["plate"] == "CAB123"
    assert types(events) == ["plate_recognized"]
    assert events[0].trigger == "http"
    # without a sensor every request is a fresh read
    again = worker.request_read("http")
    advance(1)
    assert again.result["plate"] == "CAB123"
    assert len(events) == 2


def test_http_request_for_car_already_read(setup):
    sensor = FakeSensor()
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(recognizer, sensor, mode="trigger", trigger=TRIGGER)
    sensor.active = True
    advance(2)
    request = worker.request_read("http")
    advance(0.25)
    assert request.result["plate"] == "KCA123"
    assert len(events) == 1


def test_http_request_times_out_without_plate(setup):
    recognizer = FakeRecognizer([])
    worker, events, advance = setup(recognizer, mode="trigger", trigger={**TRIGGER, "gpio_pin": None})
    request = worker.request_read("http")
    advance(6)
    assert request.result["status"] == "unrecognized"
    assert request.result["plate"] is None
    assert types(events) == ["plate_unrecognized"]


def test_pre_trigger_reads_count(setup):
    sensor = FakeSensor()
    recognizer = FakeRecognizer([make_read("KCA123")])
    worker, events, advance = setup(
        recognizer, sensor, mode="trigger", trigger={**TRIGGER, "pre_trigger_seconds": 2, "activate_delay_seconds": 0}
    )
    advance(1)  # the car approaches: read before the sensor fires
    assert events == []
    recognizer.reads = []  # the plate is hidden once the car is inside
    sensor.active = True
    advance(0.25)
    assert types(events) == ["plate_recognized"]
