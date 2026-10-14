"""Camera-shake correction: measured against star fields shifted by known amounts."""

import math

import cv2
import numpy as np
import pytest

from skyseer.config import StabiliseConfig, load_config
from skyseer.evaluate import evaluate
from skyseer.pipeline import run
from skyseer.sources import FileSource
from skyseer.stabilise import Stabiliser
from skyseer.synth import SceneSpec, _stamp, generate

FPS = 30.0


@pytest.fixture(scope="module")
def sky() -> np.ndarray:
    """A noiseless star field (float), larger than the frames cut from it."""
    rng = np.random.default_rng(7)
    img = np.full((420, 700), 25.0, np.float32)
    xs, ys = rng.uniform(0, 700, 220), rng.uniform(0, 420, 220)
    amps = rng.pareto(2.5, 220) * 25 + 20
    for x, y, amp in zip(xs, ys, amps, strict=True):
        _stamp(img, float(x), float(y), float(min(amp, 200)), 1.2)
    return img


def frames(sky: np.ndarray, shifts: np.ndarray, seed: int = 0):
    """The star field moved by each (dx, dy), plus fresh sensor noise every frame."""
    rng = np.random.default_rng(seed)
    for dx, dy in shifts:
        m = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
        img = cv2.warpAffine(sky, m, (640, 360)) + rng.normal(0, 3, (360, 640))
        yield np.clip(img, 0, 255).astype(np.uint8)


def corrections(sky, shifts, **cfg):
    stab = Stabiliser(StabiliseConfig(**cfg), FPS)
    out, moved = [], []
    for img in frames(sky, shifts):
        stab.apply(img)
        out.append(stab.correction)
        moved.append(stab.moved)
    return np.array(out), np.array(moved)


def test_wobble_is_measured_and_removed(sky):
    t = np.arange(300) / FPS
    shifts = np.stack([3 * np.sin(2 * math.pi * 2.3 * t), 2 * np.sin(2 * math.pi * 3.1 * t + 1)], 1)
    corr, moved = corrections(sky, shifts)
    settled = slice(int(FPS), None)  # the trend needs a second to find the average pointing
    error = corr[settled] - (shifts[settled] - shifts.mean(axis=0))
    assert np.sqrt((error**2).mean()) < 0.15
    assert not moved.any()


def test_a_steady_camera_is_left_untouched(sky):
    corr, moved = corrections(sky, np.zeros((120, 2)))
    assert not corr.any() and not moved.any()


def test_slow_drift_of_the_stars_is_not_corrected(sky):
    """The Earth's rotation drifts the stars steadily; the background model follows that."""
    t = np.arange(600) / FPS
    corr, _ = corrections(sky, np.stack([0.5 * t, 0.2 * t], 1))
    assert np.abs(corr).max() < 0.6  # never more than a harmless fraction of a pixel


def test_a_knock_stays_corrected(sky):
    """A sudden 4-px jump is compensated for good, keeping the original view."""
    shifts = np.zeros((300, 2))
    shifts[100:] = (4.0, -2.5)
    corr, moved = corrections(sky, shifts)
    assert not moved.any()
    assert np.abs(corr[101:] - (4.0, -2.5)).max() < 0.2


def test_a_big_jump_is_reported_as_a_camera_move(sky):
    shifts = np.zeros((120, 2))
    shifts[60:] = (30.0, 12.0)
    _, moved = corrections(sky, shifts)
    assert moved[60] and moved.sum() == 1


def test_no_stars_means_no_correction():
    stab = Stabiliser(StabiliseConfig(), FPS)
    rng = np.random.default_rng(0)
    for _ in range(40):
        img = np.clip(rng.normal(30, 3, (360, 640)), 0, 255).astype(np.uint8)
        assert stab.apply(img) is img and not stab.moved


def test_shaky_camera_end_to_end(tmp_path):
    """3 px of wind wobble: without correction it floods with false objects."""
    path = tmp_path / "shaky.mp4"
    truth = generate(SceneSpec(width=640, height=360, seconds=12, seed=3, shake_px=3.0), path)
    steady = evaluate(list(run(FileSource(path), load_config())), truth)
    assert steady.recall == 1.0 and steady.false_positives == 0, steady.confusion
    assert steady.accuracy >= 0.8, steady.confusion
    shaky = evaluate(list(run(FileSource(path), load_config(stabilise={"enabled": False}))), truth)
    assert shaky.false_positives >= 5
