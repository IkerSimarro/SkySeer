"""Real-world robustness: camera bumps / scene changes, strobes, tiny objects."""

from datetime import timedelta

import cv2
import numpy as np

from skyseer.config import load_config
from skyseer.detect import MotionDetector
from skyseer.features import flash_features
from skyseer.models import Detection, Frame, Track
from skyseer.pipeline import run
from skyseer.sources import FileSource

from .conftest import T0


def _track_with_contrast(contrast: list[float], fps: float = 30.0) -> Track:
    dets = [
        Detection(
            i,
            T0 + timedelta(seconds=i / fps),
            100.0 + i,
            50.0,
            (100 + i, 49, 3, 3),
            9.0,
            60.0,
            60.0 + c,
            contrast=c,
        )
        for i, c in enumerate(contrast)
    ]
    return Track(1, fps, dets)


def test_strobe_rhythm_is_recognised():
    rng = np.random.default_rng(0)
    n = 300  # 10 s at 30 fps
    steady = list(30 + rng.normal(0, 1.0, n))
    strobe = [c + (45 if i % 30 < 3 else 0) for i, c in enumerate(steady)]  # 1 Hz flashes
    flare = [c + (45 if 140 <= i < 144 else 0) for i, c in enumerate(steady)]  # one flare
    assert flash_features(_track_with_contrast(steady))["flash_score"] == 0.0
    s = flash_features(_track_with_contrast(strobe))
    assert s["flash_score"] > 0.8 and 0.8 < s["flash_rate_hz"] < 1.2
    assert flash_features(_track_with_contrast(flare))["flash_score"] == 0.0


def _detect_tiny(width_px: int, sensitivity: int) -> list[Detection]:
    det = MotionDetector(load_config(detector={"sensitivity": sensitivity}).detector, (200, 120))
    rng = np.random.default_rng(0)
    found: list[Detection] = []
    for i in range(40):
        img = np.clip(15 + rng.normal(0, 1.5, (120, 200)), 0, 255)  # dark, slightly noisy sky
        if i >= 20:
            img[60, 50 + 2 * i : 50 + 2 * i + width_px] = 70  # tiny object moving right
        frame = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        found = det.detect(Frame(frame, i, T0))
    return found


def test_tiny_objects_are_detected():
    """Faint 1-2 px objects must survive detection (they used to measure as area 0)."""
    found = _detect_tiny(2, sensitivity=5)  # default sensitivity: 2 px and up
    assert len(found) == 1 and abs(found[0].x - (50 + 2 * 39 + 0.5)) < 1.0
    assert len(_detect_tiny(1, sensitivity=10)) == 1  # maximum sensitivity: even 1 px


def test_camera_bump_does_not_create_fake_objects(synth_clip, tmp_path):
    """Shift the whole picture halfway through (a bumped camera): SkySeer must ignore the
    jump instead of reporting the shifted stars as moving objects."""
    path, truth = synth_clip
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(3)), int(cap.get(4)))
    out = tmp_path / "bumped.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    bump = n // 2
    for i in range(n):
        ok, frame = cap.read()
        if i >= bump:
            frame = np.roll(frame, (12, 25), axis=(0, 1))
        writer.write(frame)
    writer.release()

    stats: dict = {}
    events = list(run(FileSource(out), load_config(), stats=stats))
    assert len(stats["disturbances"]) == 1
    start, end = stats["disturbances"][0]
    assert abs(start - bump / fps) < 0.2
    pad = 1.0
    for e in events:
        t0, t1 = e.track.first_frame / fps, e.track.last_frame / fps
        assert t1 < start - pad or t0 > end + pad, "event overlaps the camera bump"
    # The jump itself must not show up as a burst of objects.
    assert len(events) <= len(truth.objects)
