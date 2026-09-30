"""Reject background stars by their shared apparent motion.

When the camera drifts (or over long exposures, as the Earth turns), stars move across
the image together: same speed, same heading, at the same time. Several tracks moving in
lockstep are therefore almost certainly stars rather than independent objects.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .config import StarFilterConfig
from .features import velocity
from .models import Track


@dataclass(slots=True)
class _Motion:
    track_id: int
    vx: float
    vy: float
    speed: float
    heading: float  # degrees
    first: int
    last: int
    cx: float
    cy: float


def _motion(track: Track) -> _Motion | None:
    if len(track.detections) < 2:
        return None
    vx, vy = velocity(track)
    pos = track.positions().mean(axis=0)
    return _Motion(
        track.id,
        vx,
        vy,
        math.hypot(vx, vy),
        math.degrees(math.atan2(vy, vx)),
        track.first_frame,
        track.last_frame,
        float(pos[0]),
        float(pos[1]),
    )


class StarFieldFilter:
    def __init__(self, config: StarFilterConfig):
        self.config = config

    def _compatible(self, a: _Motion, b: _Motion) -> bool:
        cfg = self.config
        overlap = min(a.last, b.last) - max(a.first, b.first)
        shortest = min(a.last - a.first, b.last - b.first)
        if overlap < 0 or shortest <= 0 or overlap / shortest < cfg.min_temporal_overlap:
            return False
        mean_speed = (a.speed + b.speed) / 2
        if mean_speed == 0 or abs(a.speed - b.speed) / mean_speed > cfg.velocity_tolerance:
            return False
        dh = abs(a.heading - b.heading) % 360
        if min(dh, 360 - dh) > cfg.direction_tolerance_deg:
            return False
        return math.hypot(a.cx - b.cx, a.cy - b.cy) <= cfg.max_spatial_distance

    def is_star(self, track: Track, context: Iterable[Track]) -> bool:
        """True if ``track`` moves in lockstep with enough other tracks in ``context``."""
        if not self.config.enabled:
            return False
        me = _motion(track)
        if me is None or me.speed < self.config.min_speed:
            return False
        partners = 0
        for other in context:
            if other.id == track.id:
                continue
            m = _motion(other)
            if m is not None and m.speed >= self.config.min_speed and self._compatible(me, m):
                partners += 1
                if partners >= self.config.min_group_size - 1:
                    return True
        return False

    def star_ids(self, tracks: list[Track]) -> set[int]:
        """Batch version: ids of every track that belongs to a co-moving group."""
        motions = [m for m in map(_motion, tracks) if m and m.speed >= self.config.min_speed]
        n = len(motions)
        if n < self.config.min_group_size:
            return set()
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = adj[j, i] = self._compatible(motions[i], motions[j])
        partners = adj.sum(axis=1)
        return {
            motions[i].track_id for i in range(n) if partners[i] >= self.config.min_group_size - 1
        }
