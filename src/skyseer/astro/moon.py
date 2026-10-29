"""Where the Moon is and how much of it is lit: offline, like the Sun (sun.py).

The Moon is the brightest thing a night-sky camera sees. In view, it saturates a disc
about half a degree across and lights up a glare around it where nothing can be
measured; above the horizon it brightens the whole sky, so fewer faint objects are seen.
SkySeer uses this module to know where in the picture to stop looking, and to say in the
night report how much the Moon was up.

Position: the main terms of Meeus' lunar theory (Astronomical Algorithms, 2nd ed.,
chapter 47: 60 terms in longitude and distance, 30 in latitude), checked against NASA
JPL's DE421 ephemeris in the tests. It is then turned into the observer's sky like a
satellite: as a position about 384,000 km away seen from the observer's place, so the
parallax (up to 1° from the Earth's surface) comes out right.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from sgp4.propagation import gstime

from .frames import Observer, altaz_to_vec, angular_separation_deg, apply_refraction, vec_to_altaz
from .orbits import _julian, ecef_to_enu_matrix, observer_ecef, sun_direction_teme

if TYPE_CHECKING:
    from .camera import CameraModel
    from .sky import SkyPath

MOON_RADIUS_KM = 1737.4
AU_KM = 149_597_870.7
TT_MINUS_UTC_S = 69.184  # 32.184 s + 37 leap seconds (unchanged since 2017)

# Table 47.A: multiples of D, M, M', F; longitude (1e-6 deg) and distance (1e-3 km) terms.
_LR = np.array(
    [
        (0, 0, 1, 0, 6288774, -20905355),
        (2, 0, -1, 0, 1274027, -3699111),
        (2, 0, 0, 0, 658314, -2955968),
        (0, 0, 2, 0, 213618, -569925),
        (0, 1, 0, 0, -185116, 48888),
        (0, 0, 0, 2, -114332, -3149),
        (2, 0, -2, 0, 58793, 246158),
        (2, -1, -1, 0, 57066, -152138),
        (2, 0, 1, 0, 53322, -170733),
        (2, -1, 0, 0, 45758, -204586),
        (0, 1, -1, 0, -40923, -129620),
        (1, 0, 0, 0, -34720, 108743),
        (0, 1, 1, 0, -30383, 104755),
        (2, 0, 0, -2, 15327, 10321),
        (0, 0, 1, 2, -12528, 0),
        (0, 0, 1, -2, 10980, 79661),
        (4, 0, -1, 0, 10675, -34782),
        (0, 0, 3, 0, 10034, -23210),
        (4, 0, -2, 0, 8548, -21636),
        (2, 1, -1, 0, -7888, 24208),
        (2, 1, 0, 0, -6766, 30824),
        (1, 0, -1, 0, -5163, -8379),
        (1, 1, 0, 0, 4987, -16675),
        (2, -1, 1, 0, 4036, -12831),
        (2, 0, 2, 0, 3994, -10445),
        (4, 0, 0, 0, 3861, -11650),
        (2, 0, -3, 0, 3665, 14403),
        (0, 1, -2, 0, -2689, -7003),
        (2, 0, -1, 2, -2602, 0),
        (2, -1, -2, 0, 2390, 10056),
        (1, 0, 1, 0, -2348, 6322),
        (2, -2, 0, 0, 2236, -9884),
        (0, 1, 2, 0, -2120, 5751),
        (0, 2, 0, 0, -2069, 0),
        (2, -2, -1, 0, 2048, -4950),
        (2, 0, 1, -2, -1773, 4130),
        (2, 0, 0, 2, -1595, 0),
        (4, -1, -1, 0, 1215, -3958),
        (0, 0, 2, 2, -1110, 0),
        (3, 0, -1, 0, -892, 3258),
        (2, 1, 1, 0, -810, 2616),
        (4, -1, -2, 0, 759, -1897),
        (0, 2, -1, 0, -713, -2117),
        (2, 2, -1, 0, -700, 2354),
        (2, 1, -2, 0, 691, 0),
        (2, -1, 0, -2, 596, 0),
        (4, 0, 1, 0, 549, -1423),
        (0, 0, 4, 0, 537, -1117),
        (4, -1, 0, 0, 520, -1571),
        (1, 0, -2, 0, -487, -1739),
        (2, 1, 0, -2, -399, 0),
        (0, 0, 2, -2, -381, -4421),
        (1, 1, 1, 0, 351, 0),
        (3, 0, -2, 0, -340, 0),
        (4, 0, -3, 0, 330, 0),
        (2, -1, 2, 0, 327, 0),
        (0, 2, 1, 0, -323, 1165),
        (1, 1, -1, 0, 299, 0),
        (2, 0, 3, 0, 294, 0),
        (2, 0, -1, -2, 0, 8752),
    ],
    dtype=float,
)

# Table 47.B: multiples of D, M, M', F; latitude terms (1e-6 deg).
_B = np.array(
    [
        (0, 0, 0, 1, 5128122),
        (0, 0, 1, 1, 280602),
        (0, 0, 1, -1, 277693),
        (2, 0, 0, -1, 173237),
        (2, 0, -1, 1, 55413),
        (2, 0, -1, -1, 46271),
        (2, 0, 0, 1, 32573),
        (0, 0, 2, 1, 17198),
        (2, 0, 1, -1, 9266),
        (0, 0, 2, -1, 8822),
        (2, -1, 0, -1, 8216),
        (2, 0, -2, -1, 4324),
        (2, 0, 1, 1, 4200),
        (2, 1, 0, -1, -3359),
        (2, -1, -1, 1, 2463),
        (2, -1, 0, 1, 2211),
        (2, -1, -1, -1, 2065),
        (0, 1, -1, -1, -1870),
        (4, 0, -1, -1, 1828),
        (0, 1, 0, 1, -1794),
        (0, 0, 0, 3, -1749),
        (0, 1, -1, 1, -1565),
        (1, 0, 0, 1, -1491),
        (0, 1, 1, 1, -1475),
        (0, 1, 1, -1, -1410),
        (0, 1, 0, -1, -1344),
        (1, 0, 0, -1, -1335),
        (0, 0, 3, 1, 1107),
        (4, 0, 0, -1, 1021),
        (4, 0, -1, 1, 833),
    ],
    dtype=float,
)


def _centuries_tt(when: datetime) -> float:
    jd, fr = _julian(when + timedelta(seconds=TT_MINUS_UTC_S))
    return (jd - 2451545.0 + fr) / 36525.0


def moon_ecliptic(when: datetime) -> tuple[float, float, float]:
    """Geocentric ecliptic longitude and latitude (degrees, mean equinox of date) and
    distance (km)."""
    t = _centuries_tt(when)
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t**2 + t**3 / 538841 - t**4 / 65194000
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t**2 + t**3 / 545868 - t**4 / 113065000
    m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t**2 + t**3 / 24490000
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t**2 + t**3 / 69699 - t**4 / 14712000
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t**2 - t**3 / 3526000 + t**4 / 863310000
    e = 1 - 0.002516 * t - 0.0000074 * t**2  # Earth's orbit grows rounder: scales M terms
    deg = np.array([d, m, mp, f])

    arg = np.radians(_LR[:, :4] @ deg)
    scale = e ** np.abs(_LR[:, 1])
    sl = float(np.sum(_LR[:, 4] * scale * np.sin(arg)))
    sr = float(np.sum(_LR[:, 5] * scale * np.cos(arg)))
    arg = np.radians(_B[:, :4] @ deg)
    sb = float(np.sum(_B[:, 4] * e ** np.abs(_B[:, 1]) * np.sin(arg)))

    a1, a2, a3 = np.radians([119.75 + 131.849 * t, 53.09 + 479264.290 * t, 313.45 + 481266.484 * t])
    lr, fr, mr = np.radians([lp, f, mp])
    sl += 3958 * np.sin(a1) + 1962 * np.sin(lr - fr) + 318 * np.sin(a2)
    sb += (
        -2235 * np.sin(lr)
        + 382 * np.sin(a3)
        + 175 * np.sin(a1 - fr)
        + 175 * np.sin(a1 + fr)
        + 127 * np.sin(lr - mr)
        - 115 * np.sin(lr + mr)
    )
    return (lp + sl / 1e6) % 360.0, sb / 1e6, 385000.56 + sr / 1000.0


def moon_position_teme(when: datetime) -> np.ndarray:
    """Geocentric position (km) in the equatorial frame of date (the frame SGP4 uses)."""
    lam, beta, dist = moon_ecliptic(when)
    t = _centuries_tt(when)
    eps = np.radians(23.4392911 - 0.0130042 * t)
    lam, beta = np.radians(lam), np.radians(beta)
    x = np.cos(beta) * np.cos(lam)
    y = np.cos(beta) * np.sin(lam)
    z = np.sin(beta)
    return dist * np.array(
        [x, np.cos(eps) * y - np.sin(eps) * z, np.sin(eps) * y + np.cos(eps) * z]
    )


def moon_enu(
    when: datetime, observer: Observer, refraction: bool = True
) -> tuple[np.ndarray, float]:
    """Unit vector to the Moon in the observer's local frame (east, north, up), seen from
    the observer's place (topocentric), and its distance in km. ``refraction`` lifts it
    as the atmosphere does, to match what a camera sees."""
    r = moon_position_teme(when)
    jd, fr = _julian(when)
    theta = gstime(jd + fr)
    c, s = np.cos(theta), np.sin(theta)
    ecef = np.array([c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]])
    enu = ecef_to_enu_matrix(observer) @ (ecef - observer_ecef(observer))
    dist = float(np.linalg.norm(enu))
    v = enu / dist
    return (apply_refraction(v) if refraction else v), dist


def illuminated_fraction(when: datetime) -> float:
    """Fraction of the Moon's disc that is lit (0 = new, 1 = full), seen from Earth."""
    r = moon_position_teme(when)
    dist = float(np.linalg.norm(r))
    s = sun_direction_teme(when)
    elongation = np.arccos(np.clip(r @ s / dist, -1.0, 1.0))
    phase = np.arctan2(AU_KM * np.sin(elongation), dist - AU_KM * np.cos(elongation))
    return float((1 + np.cos(phase)) / 2)


def is_waxing(when: datetime) -> bool:
    """True between new and full Moon (it rises after the Sun and is up in the evening)."""
    lam, _, _ = moon_ecliptic(when)
    s = sun_direction_teme(when)  # equatorial of date: back to ecliptic longitude
    t = _centuries_tt(when)
    eps = np.radians(23.4392911 - 0.0130042 * t)
    sun_lon = np.degrees(np.arctan2(np.cos(eps) * s[1] + np.sin(eps) * s[2], s[0]))
    return (lam - sun_lon) % 360.0 < 180.0


def phase_name(fraction: float, waxing: bool) -> str:
    if fraction < 0.03:
        return "new Moon"
    if fraction > 0.97:
        return "full Moon"
    if abs(fraction - 0.5) < 0.06:
        return "first quarter" if waxing else "last quarter"
    shape = "crescent" if fraction < 0.5 else "gibbous"
    return f"{'waxing' if waxing else 'waning'} {shape}"


@dataclass(frozen=True)
class MoonState:
    """The Moon at one moment, from one place."""

    alt: float  # degrees, apparent (refracted)
    az: float  # degrees from north through east
    fraction: float  # of the disc lit
    waxing: bool
    radius_deg: float  # apparent radius of the disc
    enu: np.ndarray  # apparent unit vector (east, north, up)

    @property
    def up(self) -> bool:
        return self.alt > -self.radius_deg

    @property
    def phase(self) -> str:
        return phase_name(self.fraction, self.waxing)

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "alt": round(self.alt, 2),
            "az": round(self.az, 2),
            "fraction_lit": round(self.fraction, 3),
            "phase": self.phase,
            "up": self.up,
        }


def moon_state(when: datetime, observer: Observer) -> MoonState:
    v, dist = moon_enu(when, observer)
    alt, az = vec_to_altaz(v)
    return MoonState(
        alt=float(alt),
        az=float(az),
        fraction=illuminated_fraction(when),
        waxing=is_waxing(when),
        radius_deg=float(np.degrees(np.arcsin(MOON_RADIUS_KM / dist))),
        enu=v,
    )


def moon_over_night(
    start: datetime,
    end: datetime,
    observer: Observer,
    camera: CameraModel | None = None,
    step: timedelta = timedelta(minutes=5),
) -> dict[str, Any]:
    """The Moon during an observing period: how lit, how long up, how high, and, with the
    calibrated ``camera`` (horizon frame), how long it was in the picture and where it
    went across the sky (for the sky map)."""
    times = [start + k * step for k in range(int((end - start) / step) + 1)] or [start]
    states = [moon_state(t, observer) for t in times]
    up = np.array([s.up for s in states])
    hours = max((end - start).total_seconds(), 0.0) / 3600  # each sample stands for 1/N
    middle = states[len(states) // 2]
    out: dict[str, Any] = {
        "fraction_lit": round(middle.fraction, 3),
        "phase": middle.phase,
        "hours_up": round(hours * float(up.mean()), 3),
        "max_alt": round(max(s.alt for s in states), 1),
        "path": [[round(s.alt, 2), round(s.az, 2)] for s in states if s.up],
    }
    if camera is not None:
        v = np.array([s.enu for s in states])
        uv, front = camera.vec_to_pixel(v)
        seen = camera.in_image(uv, front) & up
        out["minutes_in_view"] = round(60 * hours * float(seen.mean()), 1)
    return out


def moon_near_path(path: SkyPath, observer: Observer, within_deg: float) -> dict[str, Any] | None:
    """How close an object's path came to the Moon, if within ``within_deg`` of its centre
    (glare there makes brightness and position less reliable). Needs altitudes (a
    location)."""
    if path.alt is None or path.az is None or not path.times:
        return None
    n = len(path.times)
    idx = sorted({0, n // 2, n - 1})  # the Moon moves ~0.5°/min: three moments are plenty
    moon = [moon_state(path.times[i], observer) for i in idx]
    if not any(m.up for m in moon):
        return None
    where = altaz_to_vec(path.alt, path.az)
    # Each part of the path against the Moon at the nearest of the three moments.
    nearest = np.searchsorted(np.array(idx), np.arange(n), side="left").clip(0, len(idx) - 1)
    moon_v = np.array([m.enu for m in moon])[nearest]
    closest = float(np.min(angular_separation_deg(where, moon_v)))
    if closest > within_deg:
        return None
    return {"separation_deg": round(closest, 2), "fraction_lit": round(moon[0].fraction, 3)}
