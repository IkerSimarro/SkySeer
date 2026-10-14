"""Objects whose paths cross must stay two separate, straight tracks."""

import cv2
import numpy as np

from skyseer.config import load_config
from skyseer.pipeline import run
from skyseer.sources import FileSource
from skyseer.synth import _stamp


def _video(path, paths, frames=240, size=(640, 360), fps=30.0):
    rng = np.random.default_rng(0)
    w, h = size
    sky = np.full((h, w), 25.0, np.float32)
    for x, y in rng.uniform([0, 0], [w, h], (120, 2)):
        _stamp(sky, float(x), float(y), float(rng.uniform(20, 120)), 1.0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    for f in range(frames):
        img = sky + rng.normal(0, 3, (h, w)).astype(np.float32)
        for x0, y0, vx, vy, amp, start in paths:
            if f >= start:
                _stamp(img, x0 + vx * (f - start), y0 + vy * (f - start), amp, 1.3)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()


def _straight_tracks(path):
    events = [e for e in run(FileSource(path), load_config()) if e.label == "Satellite"]
    return events, [e.features["linearity"] for e in events]


def test_crossing_satellites_keep_their_identities(tmp_path):
    """An X: both cross the centre at the same moment (frame 150)."""
    path = tmp_path / "x.mp4"
    _video(path, [(170, 30, 1.0, 1.0, 90, 30), (470, 30, -1.0, 1.0, 90, 30)])
    events, linearity = _straight_tracks(path)
    assert len(events) == 2, [(e.label, e.track.duration) for e in events]
    assert min(linearity) > 0.98  # a swap at the crossing would make two bent "V" tracks
    starts = sorted(e.track.detections[0].x for e in events)
    assert abs(starts[0] - 170) < 15 and abs(starts[1] - 470) < 15


def test_overtaking_at_a_shallow_angle(tmp_path):
    """A fast satellite passes a slow one on a nearly parallel path."""
    path = tmp_path / "overtake.mp4"
    _video(path, [(60, 170, 2.2, 0.15, 90, 30), (200, 190, 0.6, -0.05, 90, 30)])
    events, linearity = _straight_tracks(path)
    assert len(events) == 2, [(e.label, e.track.duration) for e in events]
    assert min(linearity) > 0.98
    speeds = sorted(e.features["avg_speed"] for e in events)
    assert abs(speeds[0] - 0.6) < 0.15 and abs(speeds[1] - 2.2) < 0.3
