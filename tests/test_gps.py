"""GPS: NMEA sentences, the gpsd service and the location cache (no receiver needed)."""

import json
import socket
import threading
import time
from datetime import UTC, datetime

import pytest

from skyseer.astro.frames import Observer
from skyseer.gps import (
    GpsMonitor,
    NmeaReader,
    gpsd_fixes,
    load_location,
    parse_nmea,
    save_location,
)

# The classic example sentences from the NMEA 0183 documentation.
GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"


def sentence(body: str) -> str:
    total = 0
    for ch in body:
        total ^= ord(ch)
    return f"${body}*{total:02X}"


def test_classic_nmea_examples():
    gga = parse_nmea(GGA)
    assert gga["latitude"] == pytest.approx(48 + 7.038 / 60)
    assert gga["longitude"] == pytest.approx(11 + 31.0 / 60)
    assert gga["altitude_m"] == 545.4 and gga["satellites"] == 8 and gga["quality"] == 1
    rmc = parse_nmea(RMC)
    assert rmc["valid"] and rmc["date"] == "230394"


def test_corrupted_sentences_are_ignored():
    assert parse_nmea(GGA.replace("545.4", "545.5")) is None  # checksum no longer matches
    assert parse_nmea("$GPGGA,123519,4807.038,N") is None
    assert parse_nmea("garbage") is None


def test_reader_combines_position_height_and_time():
    reader = NmeaReader()
    reader.feed(RMC)
    fix = reader.feed(GGA)
    assert fix.mode == 3 and fix.satellites == 8 and fix.altitude_m == 545.4
    assert fix.time == datetime(1994, 3, 23, 12, 35, 19, tzinfo=UTC)
    assert fix.has_position and fix.observer().latitude == pytest.approx(48.1173, abs=1e-4)


def test_southern_and_western_hemispheres():
    s = sentence("GNGGA,061500.00,3352.1200,S,15112.6000,W,1,07,1.1,42.0,M,,M,,")
    fix = NmeaReader().feed(s)
    assert fix.latitude == pytest.approx(-(33 + 52.12 / 60))
    assert fix.longitude == pytest.approx(-(151 + 12.6 / 60))


def test_no_fix_yet():
    s = sentence("GPGGA,000000,,,,,0,00,99.9,,M,,M,,")
    fix = NmeaReader().feed(s)
    assert fix.mode == 0 and not fix.has_position


def _fake_gpsd(messages: list[dict]) -> int:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        conn.recv(1024)  # the ?WATCH request
        for m in messages:
            conn.sendall((json.dumps(m) + "\n").encode())
        conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1]


def test_gpsd_reports():
    port = _fake_gpsd(
        [
            {"class": "VERSION", "release": "3.25"},
            {"class": "SKY", "satellites": [{"used": True}] * 9 + [{"used": False}] * 3},
            {
                "class": "TPV",
                "mode": 3,
                "lat": 40.42,
                "lon": -3.70,
                "altMSL": 650.2,
                "time": "2026-10-21T22:00:00.000Z",
            },
        ]
    )
    fix = next(gpsd_fixes(port=port))
    assert (fix.mode, fix.satellites, fix.source) == (3, 9, "gpsd")
    assert fix.observer() == Observer(40.42, -3.70, 650.2)
    assert fix.time == datetime(2026, 10, 21, 22, tzinfo=UTC)
    assert abs(fix.clock_offset_s() - (time.time() - fix.time.timestamp())) < 5


def test_monitor_learns_and_caches_the_location(tmp_path):
    device = tmp_path / "ttyACM0"
    device.write_text(RMC + "\n" + GGA + "\n")  # a receiver's output, as a file
    cache = tmp_path / "location.json"
    gps = GpsMonitor(str(device), cache)
    gps.start()
    deadline = time.time() + 10
    while gps.location() is None and time.time() < deadline:
        time.sleep(0.05)
    gps.stop()
    assert gps.location().latitude == pytest.approx(48.1173, abs=1e-4)
    assert load_location(cache).longitude == pytest.approx(11.5167, abs=1e-4)


def test_cached_location_is_used_without_a_receiver(tmp_path):
    cache = tmp_path / "location.json"
    save_location(cache, Observer(40.42, -3.70, 650))
    gps = GpsMonitor(str(tmp_path / "missing-device"), cache)
    gps.start()
    deadline = time.time() + 10
    while gps.error is None and time.time() < deadline:
        time.sleep(0.05)
    gps.stop()
    assert gps.location() == Observer(40.42, -3.70, 650)
    status = gps.status()
    assert status["using_cached_location"] and status["error"]
