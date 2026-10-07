"""Satellite orbits: download, cache and propagate the public catalogue (SGP4).

Orbital elements come as two-line element sets (TLEs) from CelesTrak, which republishes the
US Space Force catalogue. Each TLE is a snapshot fitted at its *epoch*; SGP4 predictions are
good to about a kilometre near the epoch and drift by a few km per day after it, so recordings
must be matched against elements from (roughly) the same week.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, SatrecArray
from skyfield.sgp4lib import theta_GMST1982

from .frames import Observer

log = logging.getLogger(__name__)

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
MIN_REFRESH_HOURS = 2.0  # CelesTrak asks clients not to re-download more often than this
EARTH_RADIUS_KM = 6371.0
_WGS84_A = 6378.137
_WGS84_F = 1 / 298.257223563


@dataclass
class OrbitCatalog:
    names: list[str]
    norad: np.ndarray  # (N,) int
    epochs: np.ndarray  # (N,) Julian date (UTC) of each element set
    satrecs: list[Satrec]

    def __post_init__(self) -> None:
        self._array = SatrecArray(self.satrecs) if self.satrecs else None

    def __len__(self) -> int:
        return len(self.satrecs)

    @classmethod
    def from_tle_text(cls, text: str) -> OrbitCatalog:
        """Parse 3-line (name + 2 lines) or 2-line TLE text; keeps the newest set per object."""
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        best: dict[int, tuple[str, Satrec]] = {}
        i = 0
        while i < len(lines) - 1:
            if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
                name, l1, l2 = "", lines[i], lines[i + 1]
                i += 2
            elif (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1 ")
                and lines[i + 2].startswith("2 ")
            ):
                name = lines[i].strip().removeprefix("0 ").strip()
                l1, l2 = lines[i + 1], lines[i + 2]
                i += 3
            else:
                i += 1
                continue
            try:
                sat = Satrec.twoline2rv(l1, l2)
            except ValueError:
                continue
            prev = best.get(sat.satnum)
            # Keep a known name even if the newest element set came without a name line.
            known = name or (prev[0] if prev and not prev[0].startswith("NORAD ") else "")
            if prev is None or sat.jdsatepoch + sat.jdsatepochF > (
                prev[1].jdsatepoch + prev[1].jdsatepochF
            ):
                best[sat.satnum] = (known or f"NORAD {sat.satnum}", sat)
            elif known and prev[0].startswith("NORAD "):
                best[sat.satnum] = (known, prev[1])
        items = sorted(best.items())
        return cls(
            names=[n for _, (n, _) in items],
            norad=np.array([k for k, _ in items], dtype=int),
            epochs=np.array([s.jdsatepoch + s.jdsatepochF for _, (_, s) in items]),
            satrecs=[s for _, (_, s) in items],
        )

    @classmethod
    def from_files(cls, paths: list[Path]) -> OrbitCatalog:
        return cls.from_tle_text(
            "\n".join(Path(p).read_text(encoding="ascii", errors="replace") for p in paths)
        )

    def subset(self, idx: np.ndarray) -> OrbitCatalog:
        idx = np.asarray(idx, dtype=int)
        return OrbitCatalog(
            [self.names[i] for i in idx],
            self.norad[idx],
            self.epochs[idx],
            [self.satrecs[i] for i in idx],
        )

    def epoch_age_days(self, when: datetime) -> np.ndarray:
        return _julian(when)[0] + _julian(when)[1] - self.epochs

    def topocentric(
        self, times: list[datetime], observer: Observer
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Geometric directions of every satellite at every time, seen from ``observer``.

        Returns ``(enu_unit, range_km, r_teme_km, valid)`` with shapes (N, T, 3), (N, T),
        (N, T, 3) and (N, T). ``valid`` is False where SGP4 reported an error (e.g. decayed).
        """
        assert self._array is not None, "empty catalogue"
        jd, fr = _julian_many(times)
        err, r_teme, _ = self._array.sgp4(jd, fr)
        theta, _ = theta_GMST1982(jd, fr)
        c, s = np.cos(theta), np.sin(theta)
        # TEME → Earth-fixed: rotate by Greenwich mean sidereal time (polar motion ignored).
        x = c * r_teme[..., 0] + s * r_teme[..., 1]
        y = -s * r_teme[..., 0] + c * r_teme[..., 1]
        r_ecef = np.stack([x, y, r_teme[..., 2]], axis=-1)
        rho = r_ecef - observer_ecef(observer)
        enu = rho @ ecef_to_enu_matrix(observer).T
        rng = np.linalg.norm(enu, axis=-1)
        valid = (err == 0) & np.isfinite(rng)
        return enu / np.where(rng > 0, rng, 1.0)[..., None], rng, r_teme, valid


def observer_ecef(observer: Observer) -> np.ndarray:
    """Observer position in Earth-fixed coordinates (km), WGS84 ellipsoid."""
    lat, lon = np.radians(observer.latitude), np.radians(observer.longitude)
    e2 = _WGS84_F * (2 - _WGS84_F)
    n = _WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = observer.elevation_m / 1000.0
    return np.array(
        [
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1 - e2) + h) * np.sin(lat),
        ]
    )


def ecef_to_enu_matrix(observer: Observer) -> np.ndarray:
    lat, lon = np.radians(observer.latitude), np.radians(observer.longitude)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


def sun_direction_teme(when: datetime) -> np.ndarray:
    """Unit vector to the Sun (Astronomical Almanac low-precision formula, ~0.01°)."""
    jd, fr = _julian(when)
    n = jd + fr - 2451545.0
    g = np.radians(357.528 + 0.9856003 * n)
    lam = np.radians(280.460 + 0.9856474 * n + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * n)
    return np.array([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)])


def is_sunlit(r_teme_km: np.ndarray, when: datetime) -> np.ndarray:
    """True where a satellite is outside Earth's (cylindrical) shadow."""
    s = sun_direction_teme(when)
    along = r_teme_km @ s
    perp = np.linalg.norm(r_teme_km - along[..., None] * s, axis=-1)
    return (along > 0) | (perp > EARTH_RADIUS_KM)


def _julian(when: datetime) -> tuple[float, float]:
    when = when.astimezone(UTC)
    day = datetime(when.year, when.month, when.day, tzinfo=UTC)
    jd0 = 2440587.5 + day.timestamp() / 86400.0  # Unix epoch is JD 2440587.5
    return jd0, (when - day).total_seconds() / 86400.0


def _julian_many(times: list[datetime]) -> tuple[np.ndarray, np.ndarray]:
    pairs = [_julian(t) for t in times]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


# ---------------------------------------------------------------------------- download
def _group_path(group: str, data_dir: Path) -> Path:
    return data_dir / "tle" / f"{group}.tle"


def update_catalog(
    groups: list[str], data_dir: Path, max_age_hours: float = 24.0, force: bool = False
) -> list[Path]:
    """Download CelesTrak groups whose cached copy is missing or older than ``max_age_hours``.

    Works offline: if a download fails the cached copy is used, with a warning.
    """
    paths = []
    for group in groups:
        path = _group_path(group, data_dir)
        age_h = (time.time() - path.stat().st_mtime) / 3600 if path.exists() else None
        stale = age_h is None or age_h > max_age_hours or force
        if stale and (age_h is None or age_h > MIN_REFRESH_HOURS):
            url = CELESTRAK_URL.format(group=group)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    text = resp.read().decode("ascii", errors="replace")
                if "1 " not in text:
                    raise ValueError(f"unexpected response: {text[:80]!r}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="ascii")
                log.info("Downloaded %s orbits from CelesTrak", group)
            except (OSError, ValueError) as e:
                if not path.exists():
                    raise RuntimeError(f"Could not download orbit group '{group}': {e}") from e
                log.warning("Orbit update for '%s' failed (%s); using cached copy", group, e)
        if path.exists():
            paths.append(path)
    return paths


def load_catalog(
    groups: list[str],
    data_dir: Path,
    extra_files: list[Path] | None = None,
    max_age_hours: float = 24.0,
    download: bool = True,
) -> OrbitCatalog:
    paths = (
        update_catalog(groups, data_dir, max_age_hours)
        if download
        else [p for g in groups if (p := _group_path(g, data_dir)).exists()]
    )
    cat = OrbitCatalog.from_files(paths + list(extra_files or []))
    log.info("Loaded %d satellite orbits", len(cat))
    return cat


def epoch_datetime(jd: float) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=jd - 2440587.5)
