from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from skyseer.models import Detection, Track
from skyseer.synth import GroundTruth, SceneSpec, generate

T0 = datetime(2026, 10, 1, 22, 0, tzinfo=UTC)


def make_detection(
    frame: int, x: float, y: float, brightness: float = 120.0, size: int = 4, fps: float = 30.0
) -> Detection:
    return Detection(
        frame_index=frame,
        timestamp=T0 + timedelta(seconds=frame / fps),
        x=x,
        y=y,
        bbox=(int(x - size / 2), int(y - size / 2), size, size),
        area=float(size * size),
        mean_brightness=brightness,
        max_brightness=brightness * 1.5,
    )


def make_track(
    points: list[tuple[int, float, float]], track_id: int = 1, fps: float = 30.0, **kwargs
) -> Track:
    return Track(
        id=track_id,
        fps=fps,
        detections=[make_detection(f, x, y, fps=fps, **kwargs) for f, x, y in points],
    )


def linear_points(
    start: int, frames: int, x0: float, y0: float, vx: float, vy: float, step: int = 1
) -> list[tuple[int, float, float]]:
    return [(start + i, x0 + vx * i, y0 + vy * i) for i in range(0, frames, step)]


ASTRO_DATA = Path("data/astro")


@pytest.fixture(autouse=True, scope="session")
def _private_database(tmp_path_factory):
    """Tests never write into the real station database (session scope, so it is in
    place before any module-scoped fixture runs the CLI)."""
    patch = pytest.MonkeyPatch()
    patch.setenv("SKYSEER_DATABASE", str(tmp_path_factory.mktemp("db") / "test.db"))
    yield
    patch.undo()


@pytest.fixture(scope="session")
def catalog():
    from skyseer.astro.catalog import load_stars

    return load_stars(7.0, data_dir=ASTRO_DATA)


@pytest.fixture(scope="session")
def test_db():
    """A narrower solver database than production: builds in ~1 minute, cached on disk."""
    from skyseer.astro.catalog import build_database

    return build_database(45, 65, 6.0, data_dir=ASTRO_DATA)


@pytest.fixture(scope="session")
def synth_clip(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, GroundTruth]:
    path = tmp_path_factory.mktemp("synth") / "scene.mp4"
    truth = generate(SceneSpec(width=640, height=360, seconds=16, seed=3), path)
    return path, truth
