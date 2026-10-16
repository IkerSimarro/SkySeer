"""GPS: where the station is, and what time it really is.

The station needs both: its location decides when it is dark and which satellites it can
see, and satellite identification needs the clock right to a few seconds. A Raspberry Pi
has no clock battery, so after a power cut in a field it does not know the time until
it reaches the internet or a GPS.

Two sources, neither needing extra libraries:

- ``gpsd``, the standard Linux GPS service (``source: gpsd``): JSON over a local socket.
  It also feeds ``chrony`` for accurate system time (see deploy/pi/README.md).
- An NMEA device read directly (``source: /dev/ttyACM0``): the text sentences every GPS
  receiver sends (GGA for position and height, RMC for date and time), checksum-verified.

The last good position is cached on disk, so after a reboot the station knows where it
is straight away, before the receiver gets a fix (which can take a minute or more).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .astro.frames import Observer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fix:
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    time: datetime | None = None  # UTC, from the satellites
    mode: int = 0  # 0/1 = no fix, 2 = 2-D (no height), 3 = 3-D
    satellites: int = 0  # used in the fix
    source: str = ""
    received: float = 0.0  # time.time() when it arrived (for the clock check)

    @property
    def has_position(self) -> bool:
        return self.mode >= 2 and self.latitude is not None and self.longitude is not None

    def observer(self) -> Observer:
        assert self.latitude is not None and self.longitude is not None
        return Observer(self.latitude, self.longitude, self.altitude_m or 0.0)

    def clock_offset_s(self) -> float | None:
        """System clock minus GPS time (seconds) when the fix arrived; None without time.
        Includes the receiver's reporting delay (~0.1-0.5 s without a PPS signal)."""
        if self.time is None or not self.received:
            return None
        return round(self.received - self.time.timestamp(), 3)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["time"] = self.time.isoformat() if self.time else None
        out["clock_offset_s"] = self.clock_offset_s()
        return out


# ------------------------------------------------------------------------- NMEA
def _checksum_ok(sentence: str) -> bool:
    if "*" not in sentence:
        return False
    body, given = sentence[1:].split("*", 1)
    total = 0
    for ch in body:
        total ^= ord(ch)
    try:
        return total == int(given[:2], 16)
    except ValueError:
        return False


def _angle(value: str, hemisphere: str) -> float | None:
    """NMEA ddmm.mmmm / dddmm.mmmm -> signed decimal degrees."""
    if not value:
        return None
    dot = value.index(".") if "." in value else len(value)
    degrees = float(value[: dot - 2])
    minutes = float(value[dot - 2 :])
    sign = -1.0 if hemisphere in ("S", "W") else 1.0
    return sign * (degrees + minutes / 60.0)


def parse_nmea(line: str) -> dict[str, Any] | None:
    """Fields from a GGA or RMC sentence (any talker: GP, GN, GL, GA...), or None."""
    line = line.strip()
    if not line.startswith("$") or not _checksum_ok(line):
        return None
    fields = line[1:].split("*")[0].split(",")
    kind = fields[0][2:]
    try:
        if kind == "GGA" and len(fields) >= 10:
            quality = int(fields[6] or 0)
            return {
                "kind": "GGA",
                "utc": fields[1],
                "latitude": _angle(fields[2], fields[3]),
                "longitude": _angle(fields[4], fields[5]),
                "quality": quality,
                "satellites": int(fields[7] or 0),
                "altitude_m": float(fields[9]) if fields[9] else None,
            }
        if kind == "RMC" and len(fields) >= 10:
            return {
                "kind": "RMC",
                "utc": fields[1],
                "valid": fields[2] == "A",
                "latitude": _angle(fields[3], fields[4]),
                "longitude": _angle(fields[5], fields[6]),
                "date": fields[9],
            }
    except ValueError:
        return None
    return None


def _nmea_time(utc: str, ddmmyy: str) -> datetime | None:
    if len(utc) < 6 or len(ddmmyy) != 6:
        return None
    yy = int(ddmmyy[4:6])
    year = 1900 + yy if yy >= 80 else 2000 + yy  # two-digit years; GPS began in 1980
    day = date(year, int(ddmmyy[2:4]), int(ddmmyy[0:2]))
    seconds = float(utc[4:])
    whole = int(seconds)
    return datetime(
        day.year,
        day.month,
        day.day,
        int(utc[0:2]),
        int(utc[2:4]),
        whole,
        int(round((seconds - whole) * 1e6)) % 1000000,
        tzinfo=UTC,
    )


class NmeaReader:
    """Combines GGA (position, height, satellites) and RMC (date, time) into fixes."""

    def __init__(self) -> None:
        self.fix = Fix(source="nmea")
        self._date = ""

    def feed(self, line: str) -> Fix | None:
        """The updated fix after a useful sentence, else None."""
        msg = parse_nmea(line)
        if msg is None:
            return None
        now = time.time()
        if msg["kind"] == "RMC":
            self._date = msg["date"] or self._date
            if not msg["valid"]:
                self.fix = replace(self.fix, mode=0, received=now)
                return self.fix
            when = _nmea_time(msg["utc"], self._date)
            self.fix = replace(
                self.fix,
                latitude=msg["latitude"],
                longitude=msg["longitude"],
                time=when,
                mode=max(self.fix.mode, 2),
                received=now,
            )
            return self.fix
        mode = 0 if msg["quality"] == 0 else (3 if msg["altitude_m"] is not None else 2)
        self.fix = replace(
            self.fix,
            latitude=msg["latitude"],
            longitude=msg["longitude"],
            altitude_m=msg["altitude_m"],
            mode=mode,
            satellites=msg["satellites"],
            received=now,
        )
        return self.fix


# ------------------------------------------------------------------------- gpsd
def gpsd_fixes(host: str = "127.0.0.1", port: int = 2947, timeout: float = 10.0) -> Iterator[Fix]:
    """Fixes from gpsd's JSON stream (TPV reports, with satellite counts from SKY)."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(b'?WATCH={"enable":true,"json":true}\n')
        buffer = b""
        satellites = 0
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                try:
                    report = json.loads(raw)
                except ValueError:
                    continue
                if report.get("class") == "SKY":
                    used = [s for s in report.get("satellites", []) if s.get("used")]
                    satellites = len(used) or int(report.get("uSat", 0))
                elif report.get("class") == "TPV":
                    when = report.get("time")
                    yield Fix(
                        latitude=report.get("lat"),
                        longitude=report.get("lon"),
                        altitude_m=report.get("altMSL", report.get("alt")),
                        time=datetime.fromisoformat(when.replace("Z", "+00:00")) if when else None,
                        mode=int(report.get("mode", 0)),
                        satellites=satellites,
                        source="gpsd",
                        received=time.time(),
                    )


def nmea_fixes(device: str | Path) -> Iterator[Fix]:
    """Fixes from an NMEA device. USB receivers (/dev/ttyACM0) read like a file; a UART
    needs its baud rate set first (``stty -F /dev/serial0 9600``)."""
    reader = NmeaReader()
    with open(device, encoding="ascii", errors="replace") as fh:
        for line in fh:
            fix = reader.feed(line)
            if fix is not None:
                yield fix


# --------------------------------------------------------------------- monitor
class GpsMonitor:
    """Follows the receiver in the background; ``location()`` is always the best known
    position (the current fix, else the cached one)."""

    def __init__(
        self,
        source: str,
        cache: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 2947,
        min_satellites: int = 4,
    ):
        self.source = source
        self.cache = Path(cache) if cache else None
        self.host, self.port = host, port
        self.min_satellites = min_satellites
        self.fix: Fix | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gps", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def location(self) -> Observer | None:
        fix = self.fix
        if fix is not None and fix.has_position and fix.satellites >= self.min_satellites:
            return fix.observer()
        return load_location(self.cache) if self.cache else None

    def status(self) -> dict[str, Any]:
        fix = self.fix
        return {
            "source": self.source,
            "fix": fix.to_dict() if fix else None,
            "using_cached_location": self.location() is not None
            and not (fix and fix.has_position and fix.satellites >= self.min_satellites),
            "error": self.error,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                fixes = (
                    gpsd_fixes(self.host, self.port)
                    if self.source == "gpsd"
                    else nmea_fixes(self.source)
                )
                for fix in fixes:
                    self.error = None
                    self.fix = fix
                    if fix.has_position and fix.mode == 3 and fix.satellites >= self.min_satellites:
                        self._remember(fix)
                    if self._stop.is_set():
                        return
            except OSError as exc:  # receiver unplugged, gpsd not running...
                self.error = f"{type(exc).__name__}: {exc}"
                log.debug("GPS unavailable: %s", self.error)
            self._stop.wait(5.0)

    def _remember(self, fix: Fix) -> None:
        if self.cache is None:
            return
        old = load_location(self.cache)
        moved = (
            old is None
            or abs(old.latitude - (fix.latitude or 0)) > 1e-3
            or abs(old.longitude - (fix.longitude or 0)) > 1e-3
        )  # ~100 m
        if moved:
            save_location(self.cache, fix.observer())
            log.info("Station location from GPS: %.5f, %.5f", fix.latitude, fix.longitude)


def save_location(path: str | Path, observer: Observer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "lat": observer.latitude,
        "lon": observer.longitude,
        "elevation_m": observer.elevation_m,
        "saved": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_location(path: str | Path) -> Observer | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Observer(float(data["lat"]), float(data["lon"]), float(data.get("elevation_m", 0)))
    except (OSError, ValueError, KeyError):
        return None
