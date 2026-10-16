"""The unattended station: observes when dark, waits by day, tidies the disk.

A fake clock stands in for the real one, so a day and night pass in seconds.
"""

import json
import time
from datetime import UTC, datetime, timedelta

from skyseer.astro.frames import Observer
from skyseer.config import load_config
from skyseer.live import LiveStation
from skyseer.sources import ReplaySource
from skyseer.station import Station
from skyseer.storage import Store

MADRID = Observer(40.42, -3.70, 650)
NOON = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
MIDNIGHT = datetime(2026, 10, 21, 23, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def _config(tmp_path, **station):
    return load_config(
        storage={"live_dir": str(tmp_path / "live")},
        station={"check_interval_s": 0.05, "retry_s": 0.05, **station},
    )


def _wait(condition, seconds=60.0):
    deadline = time.time() + seconds
    while not condition():
        assert time.time() < deadline, "timed out"
        time.sleep(0.05)


def test_waits_for_dusk_then_observes(synth_clip, tmp_path):
    path, truth = synth_clip
    cfg = _config(tmp_path)
    store = Store(tmp_path / "db.sqlite")
    clock = Clock(NOON)
    nights = []

    def make_night():
        night = LiveStation(
            ReplaySource(path, speed=0),
            cfg,
            store,
            "replay",
            tmp_path / "live" / f"night{len(nights)}",
        )
        nights.append(night)
        return night

    station = Station(cfg, store, make_night, MADRID, clock)
    station.start()
    try:
        _wait(lambda: station.next_change is not None)
        status = station.status()
        assert status["mode"] == "waiting" and status["sun_altitude"] > 20
        assert (station.next_change.hour, station.next_change.minute) == (18, 25)  # dusk
        assert nights == []
        clock.now = MIDNIGHT  # night falls
        # finished, not merely "not started yet": all frames of the replay were seen
        _wait(lambda: nights and nights[0].frames == 16 * 30 and not nights[0].running)
        clock.now = NOON + timedelta(days=1)  # and the next day comes
        _wait(lambda: station.status()["mode"] == "waiting")
    finally:
        station.stop()
    # (the replay ends in seconds, so while it is still "dark" further nights start)
    first = next(s for s in store.sessions() if s["name"] == "night0")
    assert first["kind"] == "live" and first["events"] == len(truth.objects)


def test_no_location_means_observing_around_the_clock(tmp_path):
    station = Station(_config(tmp_path), Store(tmp_path / "db.sqlite"), lambda: None, None)
    assert station.is_dark()
    assert station.status()["sun_altitude"] is None


def test_a_failing_camera_is_retried(tmp_path):
    attempts = []

    def broken():
        attempts.append(1)
        raise ValueError("Could not open camera: 0")

    station = Station(
        _config(tmp_path), Store(tmp_path / "db.sqlite"), broken, MADRID, Clock(MIDNIGHT)
    )
    station.start()
    try:
        _wait(lambda: len(attempts) >= 3)
    finally:
        station.stop()
    assert "Could not open camera" in station.status()["station_error"]


def _fake_night(store, tmp_path, name, started):
    folder = tmp_path / "live" / name
    (folder / "clips").mkdir(parents=True)
    (folder / "clips" / "0001.mp4").write_bytes(b"x" * 1000)
    (folder / "run.json").write_text(json.dumps({"kind": "live"}))
    sid = store.add_session(name, "live", folder, "camera 0", started.isoformat(), {})
    store.add_events(
        sid,
        [
            {
                "id": 1,
                "start_time": started.isoformat(),
                "label": "Meteor",
                "features": {},
                "media": {"clip": "clips/0001.mp4"},
            }
        ],
    )
    return folder


def test_old_clips_are_deleted_but_records_kept(tmp_path):
    cfg = _config(tmp_path, keep_clips_days=30, min_free_gb=0)
    store = Store(tmp_path / "db.sqlite")
    old = _fake_night(store, tmp_path, "old", NOON - timedelta(days=40))
    new = _fake_night(store, tmp_path, "new", NOON - timedelta(days=2))
    analysed = tmp_path / "runs" / "recording"
    (analysed / "clips").mkdir(parents=True)  # a recording, not a live night: never touched
    store.add_session(
        "recording", "file", analysed, "video.mp4", (NOON - timedelta(days=90)).isoformat(), {}
    )
    station = Station(cfg, store, lambda: None, MADRID, Clock(NOON))
    assert station.cleanup() == ["old"]
    assert not (old / "clips").exists() and (new / "clips").exists()
    assert (analysed / "clips").exists()
    s = next(x for x in store.sessions() if x["name"] == "old")
    assert s["meta"]["clips_deleted"] and s["events"] == 1
    assert json.loads((old / "run.json").read_text())["clips_deleted"]


def test_low_disk_space_frees_the_oldest_clips_first(tmp_path):
    cfg = _config(tmp_path, keep_clips_days=365, min_free_gb=1e9)  # always "too full"
    store = Store(tmp_path / "db.sqlite")
    _fake_night(store, tmp_path, "older", NOON - timedelta(days=5))
    _fake_night(store, tmp_path, "newer", NOON - timedelta(days=1))
    station = Station(cfg, store, lambda: None, MADRID, Clock(NOON))
    assert station.cleanup() == ["older", "newer"]


def test_location_comes_from_the_gps_unless_given(tmp_path):
    from skyseer.gps import GpsMonitor, save_location

    cache = tmp_path / "location.json"
    save_location(cache, MADRID)  # the last fix, saved before a reboot
    gps = GpsMonitor(str(tmp_path / "no-receiver"), cache)
    store = Store(tmp_path / "db.sqlite")
    station = Station(_config(tmp_path), store, lambda: None, None, Clock(NOON), gps=gps)
    assert station.observer == MADRID
    assert not station.is_dark()  # noon in Madrid: no need to observe
    given = Observer(-33.9, 151.2)
    assert Station(_config(tmp_path), store, lambda: None, given, gps=gps).observer == given


def test_a_wrong_system_clock_is_reported(tmp_path):
    from skyseer.gps import Fix, GpsMonitor

    gps = GpsMonitor("gpsd")
    gps.fix = Fix(
        40.42, -3.70, 650.0, datetime.now(UTC) - timedelta(seconds=30), 3, 9, "gpsd", time.time()
    )  # the Pi thinks it is 30 s later than it is
    station = Station(
        _config(tmp_path), Store(tmp_path / "db.sqlite"), lambda: None, MADRID, gps=gps
    )
    assert "+30.0 s off GPS time" in station.status()["clock_warning"]
    gps.fix = Fix(40.42, -3.70, 650.0, datetime.now(UTC), 3, 9, "gpsd", time.time())
    assert station.status()["clock_warning"] is None
