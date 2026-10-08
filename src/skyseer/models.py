"""Core data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np


class ObjectClass(StrEnum):
    SATELLITE = "Satellite"
    METEOR = "Meteor"
    PLANE = "Plane"
    STAR = "Star"
    JUNK = "Junk"


@dataclass(slots=True)
class Frame:
    """One video frame. ``index`` is the 0-based position in the source stream."""

    image: np.ndarray
    index: int
    timestamp: datetime  # UTC wall-clock time the frame was captured


@dataclass(slots=True)
class Detection:
    """A single blob found by the motion detector, in full-resolution pixel coordinates."""

    frame_index: int
    timestamp: datetime
    x: float
    y: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float
    mean_brightness: float
    max_brightness: float
    contrast: float = 0.0  # peak brightness above the surrounding sky (reveals strobes)
    texture: float = 0.0  # spread of the surrounding sky / frame noise: ~1 clear, >3 cloud
    flux: float = 0.0  # total brightness above the sky (for magnitudes)
    saturated: bool = False  # brightest pixel clipped: true brightness is higher

    @property
    def aspect_ratio(self) -> float:
        _, _, w, h = self.bbox
        return w / h if h > 0 else 0.0


@dataclass
class Track:
    """A sequence of detections believed to belong to one physical object."""

    id: int
    fps: float  # source frame rate; speeds are expressed per source frame
    detections: list[Detection] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.detections[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame_index

    @property
    def duration(self) -> float:
        """Seconds between the first and last detection."""
        return (self.last_frame - self.first_frame) / self.fps

    @property
    def start_time(self) -> datetime:
        return self.detections[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self.detections[-1].timestamp

    def positions(self) -> np.ndarray:
        """(N, 2) array of centroid positions."""
        return np.array([(d.x, d.y) for d in self.detections], dtype=float)

    def frames(self) -> np.ndarray:
        return np.array([d.frame_index for d in self.detections], dtype=float)


@dataclass
class Event:
    """A finished, classified track: the unit SkySeer reports and stores."""

    id: int  # sequential per session, unlike track ids which also count discarded noise
    track: Track
    label: ObjectClass
    confidence: float
    features: dict[str, float]
    scores: dict[str, float]
    sky: Any = None  # astro.sky.SkyPath when the camera is calibrated
    identification: Any = None  # astro.satellites.SatelliteMatch when matched to an orbit
    identification_checked: bool = False  # True if an orbit catalogue was searched
    label_source: str = "rules"  # "rules", "starfield" or "orbit" (matched a known satellite)
    photometry: Any = None  # astro.photometry.EventPhotometry when the camera is calibrated
    shower: Any = None  # astro.showers.ShowerMatch for meteors from a known shower
    shower_checked: bool = False  # True if shower association ran (None => sporadic)
    periodicity: Any = None  # astro.lightcurve.Periodicity: regular blinking (tumbling)
    fireball: bool = False

    def to_dict(self) -> dict[str, Any]:
        t = self.track
        out = {
            "id": self.id,
            "track_id": t.id,
            "label": str(self.label),
            "label_source": self.label_source,
            "confidence": round(self.confidence, 3),
            "start_time": t.start_time.isoformat(),
            "end_time": t.end_time.isoformat(),
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "duration_s": round(t.duration, 3),
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "track": [
                {
                    "frame": d.frame_index,
                    "x": round(d.x, 1),
                    "y": round(d.y, 1),
                    "bbox": list(d.bbox),
                }
                for d in t.detections
            ],
        }
        if self.sky is not None:
            out["sky"] = self.sky.to_dict()
        if self.identification is not None:
            out["identification"] = self.identification.to_dict()
        elif self.identification_checked:
            out["identification"] = None  # searched the catalogue, nothing matched
        if self.photometry is not None:
            out["photometry"] = self.photometry.to_dict()
        if self.shower_checked:
            out["shower"] = self.shower.to_dict() if self.shower is not None else "sporadic"
        if self.periodicity is not None:
            out["periodicity"] = self.periodicity.to_dict()
        if self.fireball:
            out["fireball"] = True
        return out
