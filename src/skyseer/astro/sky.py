"""Turn a pixel track into a path across the real sky."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..models import Track
from .calibrate import Calibration
from .frames import angular_separation_deg, radec_to_vec, vec_to_altaz


@dataclass
class SkyPath:
    """Where an object was on the sky at each detection.

    RA/Dec are ICRS (J2000) degrees. Altitude/azimuth are apparent (as seen, including
    atmospheric refraction) and only present when the observer location is known.
    """

    times: list[datetime]
    ra: np.ndarray
    dec: np.ndarray
    alt: np.ndarray | None
    az: np.ndarray | None

    @property
    def vectors(self) -> np.ndarray:
        return radec_to_vec(self.ra, self.dec)

    @property
    def arc_deg(self) -> float:
        """Angle between the first and last position."""
        v = self.vectors
        return float(angular_separation_deg(v[0], v[-1]))

    @property
    def angular_speed(self) -> float:
        """Median angular speed in degrees per second.

        Measured in the horizon frame when available (that is what the camera sees);
        otherwise on the star background, which differs by at most the sky's own rotation
        (0.004 deg/s), far below any satellite's or meteor's speed.
        """
        v = self.vectors if self.alt is None else _altaz_vecs(self.alt, self.az)
        dt = np.array(
            [(b - a).total_seconds() for a, b in zip(self.times[:-1], self.times[1:], strict=True)]
        )
        if len(dt) == 0:
            return 0.0
        step = angular_separation_deg(v[:-1], v[1:])
        ok = dt > 0
        return float(np.median(step[ok] / dt[ok])) if ok.any() else 0.0

    @property
    def position_angle(self) -> float:
        """Direction of motion on the sky, degrees from celestial north through east."""
        a, b = self.vectors[0], self.vectors[-1]
        north = np.array([0.0, 0.0, 1.0]) - a[2] * a
        east = np.cross(np.array([0.0, 0.0, 1.0]), a)
        d = b - a
        return float(
            np.degrees(
                np.arctan2(d @ east / np.linalg.norm(east), d @ north / np.linalg.norm(north))
            )
            % 360
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start": {"ra": round(float(self.ra[0]), 4), "dec": round(float(self.dec[0]), 4)},
            "end": {"ra": round(float(self.ra[-1]), 4), "dec": round(float(self.dec[-1]), 4)},
            "arc_deg": round(self.arc_deg, 3),
            "angular_speed_deg_s": round(self.angular_speed, 4),
            "position_angle_deg": round(self.position_angle, 1),
        }
        if self.alt is not None and self.az is not None:
            out["start"].update(alt=round(float(self.alt[0]), 3), az=round(float(self.az[0]), 3))
            out["end"].update(alt=round(float(self.alt[-1]), 3), az=round(float(self.az[-1]), 3))
            # Up to 16 points along the path (alt, az), for drawing it on a sky map.
            k = np.unique(np.linspace(0, len(self.alt) - 1, min(16, len(self.alt))).astype(int))
            out["path"] = [[round(float(self.alt[i]), 2), round(float(self.az[i]), 2)] for i in k]
        return out


def _altaz_vecs(alt: np.ndarray, az: np.ndarray | None) -> np.ndarray:
    from .frames import altaz_to_vec

    assert az is not None
    return altaz_to_vec(alt, az)


def sky_path(track: Track, cal: Calibration) -> SkyPath:
    uv = track.positions()
    times = [d.timestamp for d in track.detections]
    ra, dec = cal.pixel_to_radec_at(uv, times)
    if cal.observer is not None:
        alt, az = vec_to_altaz(cal.pixel_to_enu(uv))
        return SkyPath(times, ra, dec, alt, az)
    return SkyPath(times, ra, dec, None, None)
