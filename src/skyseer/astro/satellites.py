"""Identify satellites: match an observed sky path against every catalogued orbit.

For each candidate orbit the predicted path is compared with the observed one in two
independent directions:

* **Cross-track** (sideways) distance. A wrong satellite on a different orbit cannot follow
  the observed line closely. This is insensitive to clock errors.
* **Along-track** (timing) offset: when the satellite actually passed each observed
  point. A consistent offset is simply the camera clock error, which is estimated and
  reported. If the offset changes along the path, the speed doesn't fit and the
  candidate is rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..config import SatelliteConfig
from .frames import Observer, altaz_to_vec, angular_separation_deg, apply_refraction
from .orbits import EARTH_RADIUS_KM, OrbitCatalog, is_sunlit
from .sky import SkyPath

log = logging.getLogger(__name__)


@dataclass
class SatelliteMatch:
    norad_id: int
    name: str
    cross_track_deg: float  # RMS sideways distance between observed and predicted paths
    clock_offset_s: float  # true time minus camera time (positive: camera clock is slow)
    timing_spread_s: float  # scatter of the per-point timing offsets
    range_km: float  # distance from the observer at mid-track
    altitude_km: float
    speed_km_s: float
    sunlit: bool
    elements_age_days: float  # observation time minus orbital-element epoch
    confidence: float
    runner_up: str | None = None
    runner_up_cross_track_deg: float | None = None

    def short_label(self) -> str:
        return f"{self.name} (NORAD {self.norad_id})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "norad_id": self.norad_id,
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "cross_track_deg": round(self.cross_track_deg, 4),
            "clock_offset_s": round(self.clock_offset_s, 2),
            "timing_spread_s": round(self.timing_spread_s, 3),
            "range_km": round(self.range_km, 1),
            "altitude_km": round(self.altitude_km, 1),
            "speed_km_s": round(self.speed_km_s, 3),
            "sunlit": self.sunlit,
            "elements_age_days": round(self.elements_age_days, 2),
            "runner_up": self.runner_up,
            "runner_up_cross_track_deg": None
            if self.runner_up_cross_track_deg is None
            else round(self.runner_up_cross_track_deg, 4),
        }


@dataclass
class _Candidate:
    index: int
    cross_rms: float
    offset: float
    spread: float


class SatelliteIdentifier:
    def __init__(self, catalog: OrbitCatalog, observer: Observer, config: SatelliteConfig):
        self.catalog = catalog
        self.observer = observer
        self.config = config
        self._warned_stale = False

    # ------------------------------------------------------------------ public
    def can_check(self, path: SkyPath) -> bool:
        """True if this path can be searched at all (location known, current elements)."""
        return (
            path.alt is not None
            and len(path.times) >= 3
            and self._elements_usable(path.times[len(path.times) // 2])
        )

    def identify(self, path: SkyPath) -> SatelliteMatch | None:
        """Best catalogued satellite for an observed path, or None if nothing fits."""
        if path.alt is None or path.az is None or len(path.times) < 3:
            return None
        cfg = self.config
        obs_t, obs_v = self._observed_points(path)
        t_mid = obs_t[len(obs_t) // 2]
        if not self._elements_usable(t_mid):
            return None

        candidates = self._shortlist(obs_t, obs_v, path.angular_speed)
        if candidates.size == 0:
            return None
        scored = sorted(
            (c for c in (self._score(i, obs_t, obs_v) for i in candidates) if c is not None),
            key=lambda c: c.cross_rms,
        )
        accepted = [
            c
            for c in scored
            if c.cross_rms <= cfg.max_cross_track_deg
            and abs(c.offset) <= cfg.max_clock_offset_s
            and c.spread <= cfg.max_timing_spread_s
        ]
        if not accepted:
            if scored:
                b = scored[0]
                log.debug(
                    "No satellite match; best %s cross %.3f° offset %.2fs spread %.2fs",
                    self.catalog.names[b.index],
                    b.cross_rms,
                    b.offset,
                    b.spread,
                )
            return None
        best = accepted[0]
        runner = accepted[1] if len(accepted) > 1 else None
        return self._describe(best, runner, t_mid)

    def predicted_path(
        self, match: SatelliteMatch, times: list[datetime]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apparent ENU directions of a matched satellite at camera ``times``.

        The measured clock offset is applied, so the prediction lines up with what the
        camera recorded. Returns (directions (T, 3), valid (T,)).
        """
        idx = np.nonzero(self.catalog.norad == match.norad_id)[0]
        true_times = [t + timedelta(seconds=match.clock_offset_s) for t in times]
        enu, valid = self._predict(idx[:1], true_times)
        return enu[0], valid[0] & (enu[0][:, 2] > 0)

    def predicted_pixels(
        self,
        match: SatelliteMatch,
        camera: Any,
        start: datetime,
        end: datetime,
        margin_s: float = 10.0,
        step_s: float = 0.25,
    ) -> np.ndarray:
        """(K, 2) pixel polyline of the predicted path from ``start - margin`` to
        ``end + margin``, clipped to the image (for drawing on the video)."""
        n = int(((end - start).total_seconds() + 2 * margin_s) / step_s) + 1
        times = [
            start - timedelta(seconds=margin_s) + timedelta(seconds=k * step_s) for k in range(n)
        ]
        enu, ok = self.predicted_path(match, times)
        uv, front = camera.vec_to_pixel(enu)
        keep = camera.in_image(uv, front & ok)
        return np.asarray(uv[keep])

    # ------------------------------------------------------------------ steps
    def _observed_points(self, path: SkyPath) -> tuple[list[datetime], np.ndarray]:
        n = len(path.times)
        idx = np.unique(np.linspace(0, n - 1, min(n, 25)).round().astype(int))
        assert path.alt is not None and path.az is not None
        vecs = altaz_to_vec(path.alt[idx], path.az[idx])  # apparent directions
        return [path.times[i] for i in idx], vecs

    def _elements_usable(self, when: datetime) -> bool:
        age = np.abs(self.catalog.epoch_age_days(when))
        median_age = float(np.median(age))
        if median_age > self.config.max_elements_age_days:
            if not self._warned_stale:
                log.warning(
                    "Orbital elements are %.0f days from the observation date; satellite "
                    "identification is skipped. For old recordings supply historical TLEs "
                    "for that date (e.g. from space-track.org) with --tle.",
                    median_age,
                )
                self._warned_stale = True
            return False
        return True

    def _predict(self, idx: np.ndarray, times: list[datetime]) -> tuple[np.ndarray, np.ndarray]:
        sub = self.catalog.subset(idx)
        enu, _, _, valid = sub.topocentric(times, self.observer)
        return apply_refraction(enu), valid  # apparent, to match what the camera saw

    def _shortlist(self, obs_t: list[datetime], obs_v: np.ndarray, speed: float) -> np.ndarray:
        """Satellites near the observed track's middle at its mid-time (all at once)."""
        cfg = self.config
        mid = len(obs_t) // 2
        enu, _, _, valid = self.catalog.topocentric([obs_t[mid]], self.observer)
        pred = apply_refraction(enu[:, 0])
        sep = angular_separation_deg(pred, obs_v[mid])
        radius = cfg.candidate_radius_deg + max(speed, 0.5) * cfg.max_clock_offset_s
        ok = valid[:, 0] & (pred[:, 2] > -0.05) & (sep < radius)
        order = np.argsort(np.where(ok, sep, np.inf))[: cfg.max_candidates]
        return order[ok[order]]

    def _score(self, index: int, obs_t: list[datetime], obs_v: np.ndarray) -> _Candidate | None:
        """Cross-track RMS and timing offset of one candidate against the observed points."""
        cfg = self.config
        pad = cfg.max_clock_offset_s + 1.0
        step = 0.05
        n = int(((obs_t[-1] - obs_t[0]).total_seconds() + 2 * pad) / step) + 1
        grid = [obs_t[0] - timedelta(seconds=pad) + timedelta(seconds=k * step) for k in range(n)]
        pred, valid = self._predict(np.array([index]), grid)
        pred, valid = pred[0], valid[0]
        if not valid.all():
            return None
        # For each observed point: nearest predicted point, refined onto the segment.
        dots = obs_v @ pred.T  # (M, n)
        k = np.argmax(dots, axis=1)
        cross = np.empty(len(obs_v))
        t_hit = np.empty(len(obs_v))
        for m, kk in enumerate(k):
            lo, hi = max(kk - 1, 0), min(kk + 1, n - 1)
            a, b = pred[lo], pred[hi]
            seg = b - a
            u = np.clip(((obs_v[m] - a) @ seg) / max(seg @ seg, 1e-18), 0.0, 1.0)
            p = a + u * seg
            p /= np.linalg.norm(p)
            cross[m] = float(angular_separation_deg(obs_v[m], p))
            t_hit[m] = (lo + u * (hi - lo)) * step - pad
        t_obs = np.array([(t - obs_t[0]).total_seconds() for t in obs_t])
        offsets = t_hit - t_obs
        # Points matched to the very ends of the window mean the path doesn't fit in time.
        if (k == 0).any() or (k == n - 1).any():
            return None
        return _Candidate(
            index,
            float(np.sqrt(np.mean(cross**2))),
            float(np.median(offsets)),
            float(np.std(offsets)),
        )

    def _describe(
        self, best: _Candidate, runner: _Candidate | None, t_mid: datetime
    ) -> SatelliteMatch:
        cfg = self.config
        true_mid = t_mid + timedelta(seconds=best.offset)
        sub = self.catalog.subset(np.array([best.index]))
        _, rng, r_teme, _ = sub.topocentric(
            [true_mid, true_mid + timedelta(seconds=1)], self.observer
        )
        r0, r1 = r_teme[0, 0], r_teme[0, 1]
        # Confidence: how well inside the acceptance limits, and how clearly ahead of the
        # runner-up (two satellites on almost the same path, e.g. a Starlink train, can
        # only be told apart when the clock is good).
        fit = 1.0 - best.cross_rms / cfg.max_cross_track_deg
        margin = (
            1.0
            if runner is None
            else min(1.0, (runner.cross_rms - best.cross_rms) / max(runner.cross_rms, 1e-6) * 3)
        )
        confidence = float(np.clip(0.5 + 0.35 * fit + 0.15 * margin, 0.0, 0.99))
        if runner is not None and runner.cross_rms < best.cross_rms * 1.2:
            confidence = min(confidence, 0.6)
        return SatelliteMatch(
            norad_id=int(self.catalog.norad[best.index]),
            name=self.catalog.names[best.index],
            cross_track_deg=best.cross_rms,
            clock_offset_s=best.offset,
            timing_spread_s=best.spread,
            range_km=float(rng[0, 0]),
            altitude_km=float(np.linalg.norm(r0) - EARTH_RADIUS_KM),
            speed_km_s=float(np.linalg.norm(r1 - r0)),
            sunlit=bool(is_sunlit(r0, true_mid)),
            elements_age_days=float(self.catalog.epoch_age_days(true_mid)[best.index]),
            confidence=confidence,
            runner_up=None if runner is None else self.catalog.names[runner.index],
            runner_up_cross_track_deg=None if runner is None else runner.cross_rms,
        )
