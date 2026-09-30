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

    def to_dict(self) -> dict[str, Any]:
        t = self.track
        return {
            "id": self.id,
            "track_id": t.id,
            "label": str(self.label),
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
