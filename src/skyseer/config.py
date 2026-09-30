"""Typed configuration. Defaults live here; YAML files override any subset of them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Section):
    sensitivity: int = Field(5, ge=1, le=10, description="1 = only bright blobs, 10 = faintest")
    process_scale: float = Field(
        1.0, gt=0, le=1, description="Downscale factor applied before detection (0.5 = half size)"
    )
    mog2_history: int = 500
    mog2_var_threshold: float = 35.0
    blur_kernel: int = 5
    morph_kernel: int = 3
    max_area_fraction: float = Field(
        0.01,
        description="Blobs larger than this fraction of the frame are ignored (clouds, lights)",
    )

    @property
    def min_area(self) -> float:
        """Minimum blob area in full-resolution pixels, derived from sensitivity."""
        return max(5.0, 50.0 - self.sensitivity * 8.0)


class TrackerConfig(_Section):
    max_distance: float = Field(
        60.0, description="Gate (px) between a track's predicted position and a detection"
    )
    new_track_distance: float = Field(
        150.0, description="Gate (px) for the second detection, before velocity is known"
    )
    max_missed_seconds: float = Field(1.0, description="Seconds unseen before a track ends")
    process_noise: float = 1.0
    measurement_noise: float = 2.0


class FilterConfig(_Section):
    min_duration: float = Field(0.3, ge=0, description="Tracks shorter than this are dropped (s)")
    max_duration: float | None = Field(None, description="Tracks longer than this are dropped (s)")
    min_detections: int = Field(3, ge=2)


class StarFilterConfig(_Section):
    enabled: bool = True
    # Star-field drift is a rigid motion, so real stars share almost identical velocities;
    # loose tolerances wrongly group unrelated objects that happen to move alike.
    min_group_size: int = 4
    velocity_tolerance: float = 0.1
    direction_tolerance_deg: float = 10.0
    min_temporal_overlap: float = 0.5
    max_spatial_distance: float = 500.0
    min_speed: float = 0.01


class RulesConfig(_Section):
    """Thresholds for the heuristic baseline classifier (px/frame, seconds)."""

    meteor_reference_speed: float = 30.0
    meteor_max_duration: float = 3.0
    satellite_duration_bands: list[tuple[float, float]] = [(3, 0.6), (18, 1.2), (25, 0.85)]
    satellite_long_factor: float = 0.65
    plane_duration_bands: list[tuple[float, float]] = [(10, 0.4), (15, 0.8)]
    plane_long_factor: float = 1.1
    blink_change_threshold: float = 0.15
    direction_change_deg: float = 30.0
    junk_base_score: float = 0.1
    stationary_speed: float = 0.5


class PipelineConfig(_Section):
    frame_skip: int = Field(1, ge=1, description="Process every Nth frame")


class Config(_Section):
    pipeline: PipelineConfig = PipelineConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    filter: FilterConfig = FilterConfig()
    star_filter: StarFilterConfig = StarFilterConfig()
    rules: RulesConfig = RulesConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load defaults, then a YAML file (if given), then keyword overrides like
    ``load_config(detector={"sensitivity": 7})``."""
    data: dict[str, Any] = Config().model_dump()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    data = _deep_merge(data, overrides)
    return Config.model_validate(data)
