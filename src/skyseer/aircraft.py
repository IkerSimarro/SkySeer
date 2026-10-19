"""Name aircraft from their ADS-B broadcasts.

Most airliners broadcast their position, altitude, speed and flight number about twice a
second on 1090 MHz (ADS-B). SkySeer gets these broadcasts either from the free OpenSky
Network online service or from a local receiver (readsb/dump1090), keeps them in a log
(``aircraft.jsonl``), works out where each aircraft was in the camera's sky, and matches
the observed path the same way satellites are matched (``astro.sky.fit_prediction``):

* **Cross-track** distance: a different aircraft can't follow the observed line.
* **Along-track** timing: a constant offset is the camera's clock error; an offset that
  changes along the path means the speed doesn't fit.

Aircraft that don't broadcast ADS-B (many light aircraft, some military) stay unnamed.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import __version__
from .astro.frames import Observer, altaz_to_vec, angular_separation_deg, apply_refraction
from .astro.sky import SkyPath, fit_prediction
from .config import AircraftConfig
from .models import Event, ObjectClass

log = logging.getLogger(__name__)

WGS84_A = 6378137.0  # m
WGS84_F = 1 / 298.257223563
_E2 = WGS84_F * (2 - WGS84_F)
FOOT = 0.3048
KNOT = 1852 / 3600


# --------------------------------------------------------------------------- geometry
def geodetic_to_ecef(lat_deg: Any, lon_deg: Any, h_m: Any) -> np.ndarray:
    """WGS84 latitude, longitude and height above the ellipsoid -> Earth-fixed metres."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    h = np.asarray(h_m, dtype=float)
    return np.stack(
        [
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1 - _E2) + h) * np.sin(lat),
        ],
        axis=-1,
    )


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Earth-fixed metres -> WGS84 latitude, longitude (degrees) and height (m)."""
    x, y, z = np.moveaxis(np.asarray(xyz, dtype=float), -1, 0)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(5):  # converges to well below a millimetre for aircraft heights
        n = WGS84_A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - _E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows: the East, North and Up unit vectors (Earth-fixed) at a place."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


# --------------------------------------------------------------------------- broadcasts
@dataclass(frozen=True, slots=True)
class AircraftState:
    """One broadcast position of one aircraft."""

    icao24: str  # the transponder's permanent address, lower-case hex
    time: float  # Unix seconds, when the position was measured
    lat: float
    lon: float
    alt_m: float  # height in metres: GNSS (geometric) when broadcast, else barometric
    geometric: bool = True
    callsign: str | None = None  # flight number, e.g. "IBE3456"
    speed_m_s: float | None = None  # ground speed
    track_deg: float | None = None  # direction of travel, clockwise from north
    vertical_m_s: float | None = None
    country: str | None = None  # country of registration (OpenSky only)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.icao24,
            "t": round(self.time, 2),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "alt": round(self.alt_m, 1),
            "geo": self.geometric,
        }
        for key, value in (
            ("cs", self.callsign),
            ("gs", self.speed_m_s),
            ("trk", self.track_deg),
            ("vr", self.vertical_m_s),
            ("cc", self.country),
        ):
            if value is not None:
                out[key] = round(value, 2) if isinstance(value, float) else value
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> AircraftState:
        return cls(
            d["id"],
            float(d["t"]),
            float(d["lat"]),
            float(d["lon"]),
            float(d["alt"]),
            bool(d.get("geo", True)),
            d.get("cs"),
            d.get("gs"),
            d.get("trk"),
            d.get("vr"),
            d.get("cc"),
        )


def parse_opensky(payload: dict[str, Any]) -> list[AircraftState]:
    """States from an OpenSky ``/states/all`` answer (aircraft on the ground are dropped)."""
    out = []
    for s in payload.get("states") or []:
        icao, cs, country, t_pos, t_last, lon, lat, baro, ground, vel, trk, vr = s[:12]
        geo = s[13] if len(s) > 13 else None
        alt = geo if geo is not None else baro
        if lat is None or lon is None or alt is None or ground:
            continue
        out.append(
            AircraftState(
                str(icao).lower(),
                float(t_pos if t_pos is not None else t_last),
                float(lat),
                float(lon),
                float(alt),
                geo is not None,
                (cs or "").strip() or None,
                None if vel is None else float(vel),
                None if trk is None else float(trk),
                None if vr is None else float(vr),
                country or None,
            )
        )
    return out


def parse_readsb(payload: dict[str, Any]) -> list[AircraftState]:
    """States from a readsb/dump1090 ``aircraft.json`` (feet and knots converted)."""
    now = float(payload.get("now", time.time()))
    out = []
    for a in payload.get("aircraft") or []:
        lat, lon, baro, geom = a.get("lat"), a.get("lon"), a.get("alt_baro"), a.get("alt_geom")
        if lat is None or lon is None or baro == "ground":
            continue
        if isinstance(geom, (int, float)):
            alt, geometric = geom * FOOT, True
        elif isinstance(baro, (int, float)):
            alt, geometric = baro * FOOT, False
        else:
            continue
        rate = a.get("geom_rate", a.get("baro_rate"))
        out.append(
            AircraftState(
                str(a["hex"]).lower().lstrip("~"),
                now - float(a.get("seen_pos", a.get("seen", 0.0))),
                float(lat),
                float(lon),
                float(alt),
                geometric,
                (a.get("flight") or "").strip() or None,
                None if a.get("gs") is None else float(a["gs"]) * KNOT,
                None if a.get("track") is None else float(a["track"]),
                None if rate is None else float(rate) * FOOT / 60,
            )
        )
    return out


class Source(Protocol):
    name: str

    def fetch(self) -> list[AircraftState]: ...


class RateLimited(RuntimeError):
    """OpenSky's daily allowance is used up (``retry_after_s`` says for how long)."""

    def __init__(self, retry_after_s: float):
        super().__init__(f"OpenSky request limit reached; retry in {retry_after_s:.0f} s")
        self.retry_after_s = retry_after_s


class OpenSkySource:
    """Current aircraft around the observer from the OpenSky Network (opensky-network.org).

    Free for non-commercial use. Without an account: 400 requests a day, positions every
    10 s. With a free account (an API client's id and secret in the environment variables
    SKYSEER_OPENSKY_CLIENT_ID and SKYSEER_OPENSKY_CLIENT_SECRET): 4,000 a day, every 5 s.
    The search box stays under 25 square degrees, which costs 1 credit per request.
    """

    name = "opensky"
    URL = "https://opensky-network.org/api/states/all"
    TOKEN_URL = (
        "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    )

    def __init__(
        self,
        observer: Observer,
        radius_km: float,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout_s: float = 15.0,
    ):
        self.observer = observer
        self.radius_km = radius_km
        self.client_id = client_id or os.environ.get("SKYSEER_OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("SKYSEER_OPENSKY_CLIENT_SECRET")
        self.timeout_s = timeout_s
        self._token: tuple[str, float] | None = None  # (token, expiry time)

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def bbox(self) -> dict[str, float]:
        """Latitude/longitude box around the observer, kept under 25 square degrees."""
        lat = self.observer.latitude
        dlat = min(self.radius_km / 111.2, 2.5)
        dlon = min(self.radius_km / (111.2 * max(np.cos(np.radians(lat)), 0.05)), 2.5)
        return {
            "lamin": round(max(lat - dlat, -90.0), 4),
            "lamax": round(min(lat + dlat, 90.0), 4),
            "lomin": round(self.observer.longitude - dlon, 4),
            "lomax": round(self.observer.longitude + dlon, 4),
        }

    def fetch(self) -> list[AircraftState]:
        url = f"{self.URL}?{urllib.parse.urlencode(self.bbox())}"
        headers = {"User-Agent": f"SkySeer/{__version__}"}
        if self.authenticated:
            headers["Authorization"] = f"Bearer {self._access_token()}"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry = float(e.headers.get("X-Rate-Limit-Retry-After-Seconds", 3600))
                raise RateLimited(retry) from e
            if e.code == 401:
                self._token = None  # expired: the next request gets a new one
            raise
        return parse_opensky(payload)

    def _access_token(self) -> str:
        if self._token is not None and time.time() < self._token[1]:
            return self._token[0]
        data = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
        ).encode()
        request = urllib.request.Request(self.TOKEN_URL, data=data, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            answer = json.load(response)
        self._token = (answer["access_token"], time.time() + answer.get("expires_in", 1800) - 60)
        return self._token[0]


class ReadsbSource:
    """A local ADS-B receiver: readsb or dump1090's ``aircraft.json`` (URL or file)."""

    name = "readsb"

    def __init__(self, location: str, timeout_s: float = 5.0):
        self.location = location
        self.timeout_s = timeout_s

    def fetch(self) -> list[AircraftState]:
        if self.location.startswith(("http://", "https://")):
            with urllib.request.urlopen(self.location, timeout=self.timeout_s) as response:
                return parse_readsb(json.load(response))
        return parse_readsb(json.loads(Path(self.location).read_text(encoding="utf-8")))


def make_source(config: AircraftConfig, observer: Observer, source: str | None = None) -> Source:
    kind = source or config.source
    if kind == "opensky":
        return OpenSkySource(observer, config.radius_km)
    if kind == "readsb":
        return ReadsbSource(config.readsb_url)
    raise ValueError(f"Unknown aircraft source {kind!r} (use opensky or readsb)")


# --------------------------------------------------------------------------- the log
class AircraftLog:
    """Every broadcast position received, per aircraft and in time order (thread-safe).

    With ``path`` each new position is also appended there as a JSON line, so a night's
    broadcasts can be matched again later. ``keep_s`` forgets positions older than that
    (to bound memory on a live station).
    """

    def __init__(
        self,
        states: Iterable[AircraftState] = (),
        path: str | Path | None = None,
        keep_s: float | None = None,
    ):
        self._by_id: dict[str, list[AircraftState]] = {}
        self._lock = threading.Lock()
        self.path = None if path is None else Path(path)
        self.keep_s = keep_s
        self.add(states, write=False)

    @classmethod
    def load(cls, path: str | Path) -> AircraftLog:
        states = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    states.append(AircraftState.from_json(json.loads(line)))
                except (ValueError, KeyError):
                    log.debug("Skipping a bad line in %s", path)
        return cls(states)

    def add(self, states: Iterable[AircraftState], write: bool = True) -> int:
        """Add positions (repeats of one already known are ignored). Returns how many
        were new."""
        new = []
        with self._lock:
            for s in states:
                track = self._by_id.setdefault(s.icao24, [])
                times = [x.time for x in track]
                i = bisect.bisect_left(times, s.time)
                if (i < len(track) and abs(track[i].time - s.time) < 0.05) or (
                    i > 0 and abs(track[i - 1].time - s.time) < 0.05
                ):
                    continue
                track.insert(i, s)
                new.append(s)
            if self.keep_s is not None and new:
                horizon = max(s.time for s in new) - self.keep_s
                for key in list(self._by_id):
                    kept = [s for s in self._by_id[key] if s.time >= horizon]
                    if kept:
                        self._by_id[key] = kept
                    else:
                        del self._by_id[key]
        if write and self.path is not None and new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                for s in new:
                    fh.write(json.dumps(s.to_json()) + "\n")
        return len(new)

    def span(self) -> tuple[float, float] | None:
        """(earliest, latest) position time, or None if empty."""
        with self._lock:
            times = [t.time for track in self._by_id.values() for t in (track[0], track[-1])]
        return (min(times), max(times)) if times else None

    def ids_between(self, t0: float, t1: float) -> list[str]:
        """Aircraft with a position between ``t0`` and ``t1``, or on both sides of it."""
        with self._lock:
            return [
                key
                for key, track in self._by_id.items()
                if track[0].time <= t1 and track[-1].time >= t0
            ]

    def states(self, icao24: str) -> list[AircraftState]:
        with self._lock:
            return list(self._by_id.get(icao24, []))

    def __len__(self) -> int:
        with self._lock:
            return sum(len(t) for t in self._by_id.values())


def trajectory(
    states: list[AircraftState], times: np.ndarray, max_gap_s: float, max_extrapolate_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Earth-fixed positions (T, 3, metres) of an aircraft at Unix ``times``, and which are
    valid. Between broadcasts up to ``max_gap_s`` apart: straight-line interpolation.
    Beyond them, up to ``max_extrapolate_s``: dead reckoning from the nearest broadcast's
    speed, heading and climb rate."""
    times = np.asarray(times, dtype=float)
    ts = np.array([s.time for s in states])
    xyz = geodetic_to_ecef(
        np.array([s.lat for s in states]),
        np.array([s.lon for s in states]),
        np.array([s.alt_m for s in states]),
    )
    out = np.zeros((len(times), 3))
    valid = np.zeros(len(times), dtype=bool)
    i = np.searchsorted(ts, times)  # times[k] lies between ts[i-1] and ts[i]
    inside = (i > 0) & (i < len(ts))
    lo, hi = np.clip(i - 1, 0, len(ts) - 1), np.clip(i, 0, len(ts) - 1)
    gap = ts[hi] - ts[lo]
    interp = inside & (gap <= max_gap_s)
    w = np.where(gap > 0, (times - ts[lo]) / np.where(gap > 0, gap, 1.0), 0.0)
    out[interp] = xyz[lo[interp]] + (xyz[hi[interp]] - xyz[lo[interp]]) * w[interp, None]
    valid |= interp
    for k in np.nonzero(~valid)[0]:  # outside the broadcasts, or across a long gap
        j = int(np.argmin(np.abs(ts - times[k])))
        s, dt = states[j], times[k] - ts[j]
        if abs(dt) > max_extrapolate_s or s.speed_m_s is None or s.track_deg is None:
            continue
        east, north, up = enu_basis(s.lat, s.lon)
        heading = np.radians(s.track_deg)
        v = s.speed_m_s * (np.sin(heading) * east + np.cos(heading) * north)
        v = v + (s.vertical_m_s or 0.0) * up
        out[k] = xyz[j] + v * dt
        valid[k] = True
    return out, valid


# --------------------------------------------------------------------------- matching
@dataclass
class AircraftMatch:
    icao24: str
    callsign: str | None
    cross_track_deg: float  # RMS sideways distance between observed and broadcast paths
    # True time minus camera time (positive: camera clock is slow). None when the aircraft
    # moves too slowly on the sky (far away) for a second of clock error to be visible.
    clock_offset_s: float | None
    timing_spread_s: float
    range_km: float  # distance from the observer at mid-track
    altitude_m: float
    speed_m_s: float  # ground speed
    heading_deg: float
    vertical_m_s: float
    geometric_altitude: bool  # altitude from GNSS (True) or barometric
    broadcast_spacing_s: float  # typical time between the positions used
    source: str  # "opensky", "readsb" or "log"
    confidence: float
    country: str | None = None
    runner_up: str | None = None
    runner_up_cross_track_deg: float | None = None

    @property
    def name(self) -> str:
        return self.callsign or self.icao24.upper()

    def short_label(self) -> str:
        return f"{self.name} (ICAO {self.icao24.upper()})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "icao24": self.icao24,
            "callsign": self.callsign,
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "cross_track_deg": round(self.cross_track_deg, 4),
            "clock_offset_s": None
            if self.clock_offset_s is None
            else round(self.clock_offset_s, 2),
            "timing_spread_s": round(self.timing_spread_s, 3),
            "range_km": round(self.range_km, 2),
            "altitude_m": round(self.altitude_m),
            "speed_m_s": round(self.speed_m_s, 1),
            "heading_deg": round(self.heading_deg, 1),
            "vertical_m_s": round(self.vertical_m_s, 1),
            "geometric_altitude": self.geometric_altitude,
            "broadcast_spacing_s": round(self.broadcast_spacing_s, 1),
            "source": self.source,
            "country": self.country,
            "runner_up": self.runner_up,
            "runner_up_cross_track_deg": None
            if self.runner_up_cross_track_deg is None
            else round(self.runner_up_cross_track_deg, 4),
        }


@dataclass
class _Candidate:
    icao24: str
    cross_rms: float
    offset: float
    spread: float


class AircraftIdentifier:
    """Matches observed sky paths against the aircraft in a broadcast log."""

    def __init__(
        self, log: AircraftLog, observer: Observer, config: AircraftConfig, source: str = "log"
    ):
        self.log = log
        self.observer = observer
        self.config = config
        self.source = source
        self._origin = geodetic_to_ecef(observer.latitude, observer.longitude, observer.elevation_m)
        self._to_enu = enu_basis(observer.latitude, observer.longitude)

    def covers(self, path: SkyPath) -> bool:
        """True once broadcasts from after the path's end are in the log, so the whole
        path can be checked (a live station waits for this)."""
        span = self.log.span()
        end = path.times[-1].timestamp() + self.config.max_clock_offset_s
        return span is not None and span[1] >= end

    def directions(self, icao24: str, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apparent ENU directions (T, 3) of an aircraft at Unix ``times`` and validity.
        Refraction is applied as for stars, which slightly overstates it for an aircraft
        inside the atmosphere: under 0.1 deg above 10 deg altitude."""
        cfg = self.config
        xyz, valid = trajectory(
            self.log.states(icao24), times, cfg.max_gap_s, cfg.max_extrapolate_s
        )
        enu = (xyz - self._origin) @ self._to_enu.T
        dist = np.linalg.norm(enu, axis=1)
        valid &= (dist > 0) & (dist < self.config.radius_km * 1000 * 1.5)
        unit = enu / np.maximum(dist, 1.0)[:, None]
        valid &= unit[:, 2] > 0
        return apply_refraction(unit), valid

    def identify(self, path: SkyPath) -> AircraftMatch | None:
        """The broadcasting aircraft that flew the observed path, or None."""
        if path.alt is None or path.az is None or len(path.times) < 3:
            return None
        cfg = self.config
        n = len(path.times)
        pick = np.unique(np.linspace(0, n - 1, min(n, 25)).round().astype(int))
        obs_t = np.array([path.times[i].timestamp() for i in pick])
        obs_v = altaz_to_vec(path.alt[pick], path.az[pick])
        # A distant aircraft crawls across the sky: then its timing along the path can't
        # be measured (a second of clock error moves it less than the noise), and the
        # along-track agreement is judged as an angle instead.
        # Net speed, first to last point: per-frame steps of a slow object are mostly noise.
        span_s = max(float(obs_t[-1] - obs_t[0]), 1e-3)
        speed = max(float(angular_separation_deg(obs_v[0], obs_v[-1])) / span_s, 1e-3)  # deg/s
        slack = min(20.0, cfg.max_cross_track_deg / speed)  # s to travel the tolerance
        pad = cfg.max_clock_offset_s + 1.0 + slack
        reach = pad + cfg.max_extrapolate_s
        ids = self.log.ids_between(obs_t[0] - reach, obs_t[-1] + reach)
        if not ids:
            return None
        mid = len(obs_t) // 2
        radius = cfg.candidate_radius_deg + max(path.angular_speed, 0.5) * cfg.max_clock_offset_s
        near = []
        for icao in ids:
            v, ok = self.directions(icao, obs_t[mid : mid + 1])
            if ok[0] and angular_separation_deg(v[0], obs_v[mid]) < radius:
                near.append(icao)
        scored = sorted(
            (c for c in (self._score(i, obs_t, obs_v, pad) for i in near) if c is not None),
            key=lambda c: c.cross_rms,
        )
        accepted = [
            c
            for c in scored
            if c.cross_rms <= cfg.max_cross_track_deg
            and (
                abs(c.offset) <= cfg.max_clock_offset_s
                or abs(c.offset) * speed <= cfg.max_cross_track_deg
            )
            and (c.spread <= cfg.max_timing_spread_s or c.spread * speed <= cfg.max_cross_track_deg)
        ]
        if not accepted:
            if scored:
                b = scored[0]
                log.debug(
                    "No aircraft match; best %s cross %.3f deg offset %.2f s spread %.2f s",
                    b.icao24,
                    b.cross_rms,
                    b.offset,
                    b.spread,
                )
            return None
        runner = accepted[1] if len(accepted) > 1 else None
        best = accepted[0]
        # The clock error is measurable if one second of it moves the aircraft by at least
        # twice the scatter of the observed points about the broadcast path.
        timed = speed * 1.0 >= 2 * max(best.cross_rms, 0.01)
        return self._describe(best, runner, float(obs_t[mid]), timed)

    def _score(
        self, icao24: str, obs_t: np.ndarray, obs_v: np.ndarray, pad: float
    ) -> _Candidate | None:
        step = 0.05
        n = int((obs_t[-1] - obs_t[0] + 2 * pad) / step) + 1
        grid = obs_t[0] - pad + step * np.arange(n)
        pred, valid = self.directions(icao24, grid)
        if not valid.all():
            return None
        fit = fit_prediction(obs_v, obs_t - obs_t[0], pred, step, pad)
        return None if fit is None else _Candidate(icao24, *fit)

    def _describe(
        self, best: _Candidate, runner: _Candidate | None, t_mid: float, timed: bool = True
    ) -> AircraftMatch:
        cfg = self.config
        states = self.log.states(best.icao24)
        t_true = t_mid + best.offset
        xyz, _ = trajectory(
            states, np.array([t_true, t_true + 1.0]), cfg.max_gap_s, cfg.max_extrapolate_s
        )
        enu = (xyz - self._origin) @ self._to_enu.T
        lat, lon, alt = ecef_to_geodetic(xyz[0])
        # Speed, heading and climb: the broadcast's own (GNSS) values when present, more
        # precise than differencing positions; otherwise from the positions.
        velocity = (xyz[1] - xyz[0]) @ enu_basis(float(lat), float(lon)).T  # local E, N, U
        closest = min(states, key=lambda s: abs(s.time - t_true))
        speed = closest.speed_m_s
        heading = closest.track_deg
        vertical = closest.vertical_m_s
        near = [s for s in states if abs(s.time - t_true) <= 60] or states
        spacing = float(np.median(np.diff([s.time for s in near]))) if len(near) > 1 else 0.0
        callsign = next((s.callsign for s in reversed(states) if s.callsign), None)
        fit = 1.0 - best.cross_rms / cfg.max_cross_track_deg
        margin = (
            1.0
            if runner is None
            else min(1.0, (runner.cross_rms - best.cross_rms) / max(runner.cross_rms, 1e-6) * 3)
        )
        confidence = float(np.clip(0.5 + 0.35 * fit + 0.15 * margin, 0.0, 0.99))
        if runner is not None and runner.cross_rms < best.cross_rms * 1.2:
            confidence = min(confidence, 0.6)
        return AircraftMatch(
            icao24=best.icao24,
            callsign=callsign,
            cross_track_deg=best.cross_rms,
            clock_offset_s=best.offset if timed else None,
            timing_spread_s=best.spread,
            range_km=float(np.linalg.norm(enu[0]) / 1000),
            altitude_m=float(alt),
            speed_m_s=float(np.hypot(velocity[0], velocity[1])) if speed is None else speed,
            heading_deg=float(np.degrees(np.arctan2(velocity[0], velocity[1])) % 360)
            if heading is None
            else heading,
            vertical_m_s=float(velocity[2]) if vertical is None else vertical,
            geometric_altitude=all(s.geometric for s in near),
            broadcast_spacing_s=spacing,
            source=self.source,
            confidence=confidence,
            country=next((s.country for s in states if s.country), None),
            runner_up=None if runner is None else runner.icao24,
            runner_up_cross_track_deg=None if runner is None else runner.cross_rms,
        )


def apply_match(event: Event, match: AircraftMatch | None) -> None:
    """Record an aircraft check on an event. A confident match makes it a Plane: a
    broadcasting aircraft really was there, which outranks the motion rules and any
    satellite that happened to fit."""
    event.aircraft_checked = True
    event.aircraft = match
    if match is not None and match.confidence >= 0.5:
        event.label, event.label_source = ObjectClass.PLANE, "adsb"
        event.confidence = match.confidence
        event.identification, event.identification_checked = None, False


# --------------------------------------------------------------------------- live feed
class AircraftFeed:
    """Keeps an :class:`AircraftLog` up to date from a live source, in a background thread.

    With ``active`` (a function saying whether something is crossing the view) and
    ``config.active_only``, OpenSky is asked only while it matters, to stay within its free
    daily allowance; a local receiver is always read (it costs nothing).
    """

    def __init__(
        self,
        source: Source,
        log_: AircraftLog,
        config: AircraftConfig,
        active: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.source = source
        self.log = log_
        self.config = config
        self.active = active
        self.clock = clock
        self.requests_today = 0
        self._day = datetime.now(UTC).date()
        self.last_update: float | None = None
        self.in_range = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused_until = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="aircraft", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _wanted(self) -> bool:
        if self.source.name != "opensky" or not self.config.active_only or self.active is None:
            return True
        return self.active()

    def _run(self) -> None:
        while not self._stop.wait(self.config.poll_s):
            if self._wanted():
                self.poll()

    def poll(self) -> int:
        """Ask the source once; returns how many new positions arrived."""
        now = self.clock()
        today = datetime.fromtimestamp(now, UTC).date()
        if today != self._day:
            self._day, self.requests_today = today, 0
        if now < self._paused_until:
            return 0
        if (
            self.source.name == "opensky"
            and self.requests_today >= self.config.max_requests_per_day
        ):
            self.error = "today's OpenSky request budget is used (max_requests_per_day)"
            return 0
        try:
            self.requests_today += 1
            states = self.source.fetch()
        except RateLimited as e:
            self._paused_until = now + e.retry_after_s
            self.error = str(e)
            log.warning("%s", e)
            return 0
        except Exception as e:  # a network hiccup must not stop the station
            self.error = f"{type(e).__name__}: {e}"
            log.warning("Aircraft feed: %s", self.error)
            return 0
        self.error = None
        self.last_update = now
        self.in_range = len({s.icao24 for s in states})
        return self.log.add(states)

    def status(self) -> dict[str, Any]:
        return {
            "source": self.source.name,
            "authenticated": bool(getattr(self.source, "authenticated", False)),
            "last_update": None
            if self.last_update is None
            else datetime.fromtimestamp(self.last_update, UTC).isoformat(),
            "aircraft": self.in_range,
            "requests_today": self.requests_today,
            "error": self.error,
        }
