"""Heuristic baseline classifier.

Each class gets a hand-designed score from the track features; the highest wins. The
``confidence`` is the winning score's share of all scores. It expresses how clearly one
rule dominates, **not** a calibrated probability. It is kept as a transparent baseline
to compare the learned model against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RulesConfig
from ..models import ObjectClass


@dataclass(slots=True)
class Classification:
    label: ObjectClass
    confidence: float
    scores: dict[str, float]


def _band(value: float, bands: list[tuple[float, float]], above: float) -> float:
    """Piecewise-constant factor: the first (upper_bound, factor) whose bound exceeds value."""
    for upper, factor in bands:
        if value < upper:
            return factor
    return above


class RuleClassifier:
    def __init__(self, config: RulesConfig):
        self.config = config

    def scores(self, f: dict[str, float]) -> dict[str, float]:
        c = self.config
        duration, linearity = f["duration"], f["linearity"]
        speed = f["speed_frac_per_s"]  # image diagonals per second: resolution-independent
        steady = f["size_consistency"] * f["brightness_consistency"]
        flash = f["flash_score"]

        # Satellite: steady speed, straight line, constant brightness, no strobes.
        satellite = (
            f["speed_consistency"]
            * linearity
            * steady
            * _band(duration, c.satellite_duration_bands, c.satellite_long_factor)
            * (1.0 - 0.85 * flash)
        )

        # Meteor: fast, straight, brief, often bright.
        speed_factor = min(speed / c.meteor_reference_speed, 3.0)
        brief = 1.0 / max(duration, 0.1) if duration < c.meteor_max_duration else 0.3
        bright = 1.0 + min(f["max_brightness"] / 255.0, 0.5)
        # Meteors are straight (a curved fast streak is an insect or bat near the lens) and
        # leave a continuous trail rather than a handful of flickering dots.
        straight = float(np.clip(1.0 - (f["path_deviation"] - 0.01) / 0.04, 0.0, 1.0))
        enough = float(np.clip((f["detection_count"] - 3.0) / 4.0, 0.0, 1.0))
        meteor = speed_factor * linearity * brief * bright * straight * enough

        # Plane: moves like a satellite but its strobes flash at a steady rhythm.
        plane = (
            f["speed_consistency"]
            * linearity
            * _band(duration, c.plane_duration_bands, c.plane_long_factor)
            * (c.plane_base_score + 1.2 * flash)
        )

        # Cloud texture: bright fragments that "move" inside clouds are not objects. Real
        # objects crossing clear sky have cloudiness ~1.
        clean = min(1.0, (c.cloud_texture / max(f["cloudiness"], 1e-6)) ** 2)
        satellite, meteor, plane = satellite * clean, meteor * clean, plane * clean

        junk = c.junk_base_score / max(clean, 0.05)
        if speed < c.stationary_speed:
            junk *= 5.0  # hot pixels, twinkling stars, stationary noise
        elif speed > c.meteor_reference_speed and linearity > 0.85 and duration < 1.5:
            meteor *= 1.8
        elif f["speed_consistency"] < 0.3 or linearity < 0.3:
            junk *= 1.5

        return {
            ObjectClass.SATELLITE: satellite,
            ObjectClass.METEOR: meteor,
            ObjectClass.PLANE: plane,
            ObjectClass.JUNK: junk,
        }

    def classify(self, features: dict[str, float]) -> Classification:
        scores = self.scores(features)
        label = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        confidence = scores[label] / total if total > 0 else 0.0
        return Classification(
            ObjectClass(label), confidence, {str(k): v for k, v in scores.items()}
        )
