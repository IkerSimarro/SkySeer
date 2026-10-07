"""Coordinate frames.

Directions are 3-D unit vectors. Two frames matter:

* **ICRS** (≈ J2000 equatorial): fixed relative to the stars. Star catalogues and the
  plate solver work here.
* **ENU** (East, North, Up) at the observer: fixed relative to the ground. A camera bolted
  to a station keeps the same ENU pointing all night, while the stars drift through it at
  15 degrees per hour; so SkySeer stores its camera calibration in ENU.

ICRS → ENU uses precession-nutation and Earth rotation from Skyfield. Polar motion and
annual aberration (< 21 arcsec) are ignored: they are far below a wide-field camera's pixel
scale (~100 arcsec/px).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
from skyfield.api import load

ArrayLike = np.ndarray


@dataclass(frozen=True, slots=True)
class Observer:
    latitude: float  # degrees, north positive
    longitude: float  # degrees, east positive
    elevation_m: float = 0.0


@lru_cache(maxsize=1)
def timescale() -> Any:
    return load.timescale(builtin=True)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def icrs_to_enu_matrix(when: datetime, observer: Observer) -> np.ndarray:
    """3x3 rotation taking ICRS unit vectors to the observer's local ENU frame."""
    t = timescale().from_datetime(when)
    # ICRS -> true equator & equinox of date -> Earth-fixed (rotate by sidereal time).
    to_date = np.asarray(t.M)
    gast = np.radians(float(t.gast) * 15.0)
    to_ecef = _rot_z(gast) @ to_date
    lat, lon = np.radians(observer.latitude), np.radians(observer.longitude)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    ecef_to_enu = np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )
    return ecef_to_enu @ to_ecef


def radec_to_vec(ra_deg: ArrayLike | float, dec_deg: ArrayLike | float) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def vec_to_radec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    ra = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    return ra, dec


def altaz_to_vec(alt_deg: ArrayLike | float, az_deg: ArrayLike | float) -> np.ndarray:
    """Azimuth measured from North through East (compass convention)."""
    alt, az = np.radians(alt_deg), np.radians(az_deg)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)


def vec_to_altaz(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    alt = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360.0
    return alt, az


def refraction_deg(true_alt_deg: ArrayLike | float) -> np.ndarray:
    """Atmospheric refraction (Saemundsson formula, 10 °C, 1010 hPa), in degrees.

    Refraction lifts objects near the horizon by up to ~0.5 degrees, so it matters for a
    wide-field camera that sees low altitudes.
    """
    h = np.maximum(np.asarray(true_alt_deg, dtype=float), -1.0)
    return 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0


def apply_refraction(v_enu: np.ndarray) -> np.ndarray:
    """Geometric ENU directions → apparent (refracted) directions."""
    alt, az = vec_to_altaz(v_enu)
    return altaz_to_vec(alt + refraction_deg(alt), az)


def remove_refraction(v_enu: np.ndarray) -> np.ndarray:
    """Apparent ENU directions → geometric directions (fixed-point inversion)."""
    alt_app, az = vec_to_altaz(v_enu)
    alt = alt_app.copy()
    for _ in range(4):
        alt = alt_app - refraction_deg(alt)
    return altaz_to_vec(alt, az)


def angular_separation_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors (numerically stable for tiny angles)."""
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.degrees(np.arctan2(cross, dot))
