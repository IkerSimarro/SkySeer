"""Meteor showers: which shower (if any) did a meteor come from?

Meteors from one shower travel on parallel paths, so seen from the ground they appear to
fan out from a single point, the *radiant*. A meteor is associated with a shower when its
path, extended backwards across the sky, passes close to that shower's radiant, it
moves away from it, the shower is active on that date, the radiant is above the horizon,
and (when the location is known) its angular speed matches the shower's entry speed.
Everything else is *sporadic*.

Shower data: IAU Meteor Data Center, established showers (bundled; see data/showers.json
for the source and citation).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

from ..config import ShowerConfig
from .frames import (
    Observer,
    angular_separation_deg,
    icrs_to_enu_matrix,
    radec_to_vec,
    vec_to_altaz,
)
from .sky import SkyPath

EARTH_RADIUS_KM = 6371.0
ESCAPE_V2 = 125.4  # (km/s)^2: square of Earth's escape velocity at the top of the atmosphere


def solar_longitude(when: datetime) -> float:
    """Apparent ecliptic longitude of the Sun in degrees, referred to the J2000 equinox.

    Showers are dated by solar longitude because it repeats with the Earth's orbit rather
    than the calendar. Low-precision solar theory (~0.01°) plus a precession correction
    from the equinox of date to J2000 (~0.014° per year).
    """
    when = when.astimezone(UTC)
    jd = 2440587.5 + when.timestamp() / 86400.0
    n = jd - 2451545.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam_of_date = 280.460 + 0.9856474 * n + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)
    precession = 1.396971 * (n / 36525.0)  # degrees from J2000 to the equinox of date
    return (lam_of_date - precession) % 360.0


def _wrap(deg: float) -> float:
    """Wrap an angle difference to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Shower:
    code: str
    iau_no: int
    name: str
    lsun_peak: float
    lsun_begin: float
    lsun_end: float
    ra: float
    dec: float
    dra: float  # radiant drift per degree of solar longitude
    ddec: float
    vg: float  # geocentric velocity, km/s
    members: int

    def active(self, lsun: float) -> bool:
        span = (self.lsun_end - self.lsun_begin) % 360.0
        return (lsun - self.lsun_begin) % 360.0 <= span

    def radiant(self, lsun: float) -> tuple[float, float]:
        """Radiant RA/Dec (J2000 degrees) on a given solar longitude, with daily drift."""
        d = _wrap(lsun - self.lsun_peak)
        return (self.ra + self.dra * d) % 360.0, self.dec + self.ddec * d

    @property
    def v_inf(self) -> float:
        """Speed entering the atmosphere: Earth's gravity adds to the geocentric speed."""
        return math.sqrt(self.vg**2 + ESCAPE_V2)


@lru_cache(maxsize=1)
def load_showers() -> tuple[Shower, ...]:
    data = json.loads(
        resources.files("skyseer.astro").joinpath("data/showers.json").read_text(encoding="utf-8")
    )
    return tuple(
        Shower(
            s["code"],
            s["iau_no"],
            s["name"],
            s["solar_longitude_peak"],
            s["solar_longitude_begin"],
            s["solar_longitude_end"],
            s["ra"],
            s["dec"],
            s["dra_per_deg"],
            s["ddec_per_deg"],
            s["vg_km_s"],
            s["members"],
        )
        for s in data["showers"]
    )


def active_showers(when: datetime, min_members: int = 0) -> list[Shower]:
    lsun = solar_longitude(when)
    return [s for s in load_showers() if s.active(lsun) and s.members >= min_members]


@dataclass
class ShowerMatch:
    code: str
    name: str
    iau_no: int
    radiant_ra: float
    radiant_dec: float
    radiant_offset_deg: float  # closest approach of the backward path to the radiant
    radiant_distance_deg: float  # angle between radiant and the meteor
    radiant_elevation_deg: float | None
    speed_ratio: float | None  # observed / expected angular speed (needs the location)
    confidence: float

    def short_label(self) -> str:
        return f"{self.name} ({self.code})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "iau_no": self.iau_no,
            "radiant_ra": round(self.radiant_ra, 2),
            "radiant_dec": round(self.radiant_dec, 2),
            "radiant_offset_deg": round(self.radiant_offset_deg, 2),
            "radiant_distance_deg": round(self.radiant_distance_deg, 1),
            "radiant_elevation_deg": None
            if self.radiant_elevation_deg is None
            else round(self.radiant_elevation_deg, 1),
            "speed_ratio": None if self.speed_ratio is None else round(self.speed_ratio, 2),
            "confidence": round(self.confidence, 2),
        }


def meteor_height_km(v_inf: float) -> float:
    """Typical height of a meteor's luminous path: faster meteors burn up higher
    (~110 km at 70 km/s, ~90 km at 25 km/s)."""
    return 80.0 + 0.43 * v_inf


def range_to_height_km(elevation_deg: float, height_km: float) -> float:
    """Distance from the observer to a point at ``height_km`` seen at ``elevation_deg``
    (spherical Earth)."""
    e = math.radians(max(elevation_deg, 0.5))
    r = EARTH_RADIUS_KM
    return math.sqrt((r + height_km) ** 2 - (r * math.cos(e)) ** 2) - r * math.sin(e)


def expected_angular_speed(
    v_inf: float, radiant_distance_deg: float, elevation_deg: float
) -> float:
    """Angular speed (deg/s) of a meteor at ``radiant_distance_deg`` from its radiant: only
    the velocity component across the line of sight is seen (∝ sin D), divided by range."""
    rng = range_to_height_km(elevation_deg, meteor_height_km(v_inf))
    return math.degrees(v_inf * math.sin(math.radians(radiant_distance_deg)) / rng)


class ShowerAssociator:
    """Associates meteor sky paths with active showers (or calls them sporadic)."""

    def __init__(self, config: ShowerConfig, observer: Observer | None):
        self.config = config
        self.observer = observer

    def associate(self, path: SkyPath) -> ShowerMatch | None:
        c = self.config
        vecs = path.vectors
        start, end = vecs[0], vecs[-1]
        arc = float(angular_separation_deg(start, end))
        if len(vecs) < 3 or arc < c.min_arc_deg:
            return None
        pole = np.cross(start, end)
        pole /= np.linalg.norm(pole)
        mid_time = path.times[len(path.times) // 2]
        lsun = solar_longitude(mid_time)
        mid = vecs[len(vecs) // 2]
        enu = None if self.observer is None else icrs_to_enu_matrix(mid_time, self.observer)

        best: tuple[float, ShowerMatch] | None = None
        for shower in active_showers(mid_time, c.min_members):
            ra, dec = shower.radiant(lsun)
            r = radec_to_vec(ra, dec)
            offset = float(np.degrees(abs(math.asin(np.clip(r @ pole, -1.0, 1.0)))))
            # Pointing errors grow when a short path is extended far back to the radiant.
            dist = float(angular_separation_deg(r, mid))
            tol = min(
                c.max_radiant_offset_deg + dist * c.direction_error_deg / max(arc, 0.1),
                c.max_offset_cap_deg,
            )
            if offset > tol:
                continue
            # Meteors fly away from the radiant, and never right on top of it or beyond 90°.
            if not (
                float(angular_separation_deg(r, start)) < float(angular_separation_deg(r, end))
            ):
                continue
            if not (c.min_radiant_distance_deg <= dist <= c.max_radiant_distance_deg):
                continue
            elev = None
            ratio = None
            if enu is not None:
                elev = float(vec_to_altaz(r @ enu.T)[0])
                if elev < c.min_radiant_elevation_deg:
                    continue
                m_alt = float(vec_to_altaz(mid @ enu.T)[0])
                expected = expected_angular_speed(shower.v_inf, dist, m_alt)
                ratio = path.angular_speed / expected if expected > 0 else None
                if ratio is None or not (c.min_speed_ratio <= ratio <= c.max_speed_ratio):
                    continue
            # Lower score = better: offset within tolerance, speed agreement.
            score = offset / tol + (abs(math.log(ratio)) if ratio else 0.5)
            fit = max(0.0, 1.0 - offset / tol)
            speed_fit = (
                1.0 - min(1.0, abs(math.log(ratio)) / math.log(c.max_speed_ratio)) if ratio else 0.5
            )
            confidence = float(np.clip(0.35 + 0.35 * fit + 0.3 * speed_fit, 0.0, 0.95))
            match = ShowerMatch(
                shower.code,
                shower.name,
                shower.iau_no,
                ra,
                dec,
                offset,
                dist,
                elev,
                ratio,
                confidence,
            )
            if best is None or score < best[0]:
                best = (score, match)
        return None if best is None else best[1]
