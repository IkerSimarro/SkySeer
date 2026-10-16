"""Where the Sun is, for scheduling the station's nights.

Works offline (no ephemeris files): the Astronomical Almanac's low-precision solar
formula (about 0.01°, far better than twilight needs) rotated into the observer's
horizon frame with Greenwich sidereal time.

Twilight is defined by the Sun's altitude below the horizon: -6° civil, -12° nautical,
-18° astronomical. Around nautical twilight the sky is dark enough for stars and
satellites, and satellites are still sunlit high above, so the best passes come early.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from sgp4.propagation import gstime

from .frames import Observer
from .orbits import _julian, ecef_to_enu_matrix, sun_direction_teme


def sun_altitude(when: datetime, observer: Observer) -> float:
    """The Sun's altitude in degrees (refraction and parallax ignored: < 0.6°)."""
    jd, fr = _julian(when)
    theta = gstime(jd + fr)
    s = sun_direction_teme(when)  # equatorial frame of date
    c, si = np.cos(theta), np.sin(theta)
    ecef = np.array([c * s[0] + si * s[1], -si * s[0] + c * s[1], s[2]])
    up = float((ecef_to_enu_matrix(observer) @ ecef)[2])
    return float(np.degrees(np.arcsin(np.clip(up, -1.0, 1.0))))


def is_dark(when: datetime, observer: Observer, sun_altitude_deg: float = -12.0) -> bool:
    return sun_altitude(when, observer) < sun_altitude_deg


def next_change(
    now: datetime,
    observer: Observer,
    sun_altitude_deg: float = -12.0,
    horizon: timedelta = timedelta(hours=48),
) -> datetime | None:
    """When darkness next begins or ends (to within a few seconds), or None if it won't
    within ``horizon`` (polar day or night)."""
    dark = is_dark(now, observer, sun_altitude_deg)
    step = timedelta(minutes=5)
    t = now
    while t - now < horizon:
        nxt = t + step
        if is_dark(nxt, observer, sun_altitude_deg) != dark:
            lo, hi = t, nxt  # bisect the crossing
            while hi - lo > timedelta(seconds=5):
                mid = lo + (hi - lo) / 2
                if is_dark(mid, observer, sun_altitude_deg) == dark:
                    lo = mid
                else:
                    hi = mid
            return hi
        t = nxt
    return None


def dark_windows(
    start: datetime, end: datetime, observer: Observer, sun_altitude_deg: float = -12.0
) -> list[tuple[datetime, datetime]]:
    """The dark periods between ``start`` and ``end``."""
    windows = []
    t = start
    dark_since = start if is_dark(start, observer, sun_altitude_deg) else None
    while t < end:
        change = next_change(t, observer, sun_altitude_deg, end - t)
        if change is None:
            break
        if dark_since is None:
            dark_since = change
        else:
            windows.append((dark_since, change))
            dark_since = None
        t = change + timedelta(seconds=1)
    if dark_since is not None:
        windows.append((dark_since, end))
    return windows
