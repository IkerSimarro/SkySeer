"""Phase 1 completion: re-calibration, sky drift, sky paths and live auto-calibration."""

import time as wallclock
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from skyseer.astro.autocal import AutoCalibrator
from skyseer.astro.calibrate import calibrate
from skyseer.astro.frames import Observer, angular_separation_deg, radec_to_vec, vec_to_radec
from skyseer.astro.simulate import camera_looking_at, render_starfield
from skyseer.astro.sky import sky_path
from skyseer.config import AstroConfig
from skyseer.models import Detection, Frame, Track

from .conftest import ASTRO_DATA

OBS = Observer(40.4168, -3.7038, 650)
T0 = datetime(2026, 8, 12, 22, 30, tzinfo=UTC)


def _truth_radec(truth, uv, when, observer=OBS):
    """RA/Dec behind pixels of a ground-fixed (ENU) camera at a given time."""
    from skyseer.astro.frames import icrs_to_enu_matrix, remove_refraction

    v = remove_refraction(truth.pixel_to_vec(uv)) @ icrs_to_enu_matrix(when, observer)
    return vec_to_radec(v)


@pytest.fixture(scope="module")
def first_cal(catalog, test_db):
    truth = camera_looking_at(55, 120, 3, 1920, 1080, 55, k1=-0.05)
    img = render_starfield(truth, OBS, T0, catalog)
    return truth, calibrate(
        img, T0, OBS, fov_estimate=55, fov_max_error=10, database=test_db, catalog=catalog
    )


def test_recalibration_from_hint_is_fast_and_accurate(first_cal, catalog, test_db):
    truth, cal0 = first_cal
    later = T0 + timedelta(minutes=40)  # stars have moved ~10 degrees through the frame
    img = render_starfield(truth, OBS, later, catalog, seed=5)
    t = wallclock.perf_counter()
    cal = calibrate(img, later, OBS, database=test_db, catalog=catalog, hint=cal0)
    elapsed = wallclock.perf_counter() - t
    assert cal.solver["method"] == "hint"
    assert elapsed < 3.0
    assert float(angular_separation_deg(cal.camera.boresight, truth.boresight)) < 30 / 3600


def test_moved_camera_falls_back_to_full_solve(first_cal, catalog, test_db):
    _, cal0 = first_cal
    moved = camera_looking_at(40, 250, -5, 1920, 1080, 55, k1=-0.05)
    img = render_starfield(moved, OBS, T0, catalog, seed=6)
    cal = calibrate(
        img,
        T0,
        OBS,
        fov_estimate=55,
        fov_max_error=10,
        database=test_db,
        catalog=catalog,
        hint=cal0,
    )
    assert cal.solver["method"] == "lost-in-space"
    assert float(angular_separation_deg(cal.camera.boresight, moved.boresight)) < 30 / 3600


def test_sky_coordinates_follow_earth_rotation_without_location(catalog, test_db):
    """Calibrated without knowing where it is, the camera still reports correct RA/Dec an
    hour later, because the sky's rotation is applied."""
    truth = camera_looking_at(60, 200, 0, 1920, 1080, 55, k1=-0.04)
    cal = calibrate(
        render_starfield(truth, OBS, T0, catalog),
        T0,
        None,
        fov_estimate=55,
        fov_max_error=10,
        database=test_db,
        catalog=catalog,
    )
    uv = np.array([[300.0, 200.0], [960.0, 540.0], [1700.0, 900.0]])
    later = T0 + timedelta(hours=1)
    ra, dec = cal.pixel_to_radec(uv, later)
    ra_t, dec_t = _truth_radec(truth, uv, later)
    sep = angular_separation_deg(radec_to_vec(ra, dec), radec_to_vec(ra_t, dec_t))
    # Refraction can't be modelled without a location: allow a few arcminutes.
    assert sep.max() < 0.1
    # Vectorised per-point version agrees.
    ra2, dec2 = cal.pixel_to_radec_at(uv, [later] * 3)
    assert np.allclose(radec_to_vec(ra2, dec2), radec_to_vec(ra, dec), atol=1e-9)


def test_sky_path_speed_and_direction(first_cal):
    truth, cal = first_cal
    # An object moving 1 degree/s straight "up" the image for 5 s (10 detections/s).
    times = [T0 + timedelta(seconds=0.1 * i) for i in range(51)]
    start = truth.pixel_to_vec(np.array([[960.0, 900.0]]))[0]
    up = truth.pixel_to_vec(np.array([[960.0, 100.0]]))[0]
    axis = np.cross(start, up)
    axis /= np.linalg.norm(axis)
    dets = []
    for i, t in enumerate(times):
        ang = np.radians(0.1 * i)
        v = (
            start * np.cos(ang)
            + np.cross(axis, start) * np.sin(ang)
            + axis * (axis @ start) * (1 - np.cos(ang))
        )
        uv, _ = truth.vec_to_pixel(v[None])
        dets.append(Detection(i, t, float(uv[0, 0]), float(uv[0, 1]), (0, 0, 2, 2), 4, 100, 150))
    path = sky_path(Track(1, 10.0, dets), cal)
    assert path.angular_speed == pytest.approx(1.0, rel=0.02)
    assert path.arc_deg == pytest.approx(5.0, rel=0.02)
    assert path.alt is not None and path.alt[-1] > path.alt[0]  # moving upwards
    d = path.to_dict()
    assert {"start", "end", "angular_speed_deg_s", "position_angle_deg"} <= d.keys()


def test_auto_calibrator_solves_in_background(catalog, test_db, tmp_path):
    truth = camera_looking_at(65, 30, 0, 1920, 1080, 55, k1=-0.05)
    img = render_starfield(truth, OBS, T0, catalog)
    cfg = AstroConfig(
        fov_estimate=55,
        data_dir=str(ASTRO_DATA),
        database_min_fov=45,
        database_max_fov=65,
        database_max_mag=6.0,
        calibration_seconds=1.0,
    )
    cache = tmp_path / "live_cal.json"
    auto = AutoCalibrator(cfg, OBS, fps=15, cache_path=cache)
    for i in range(20):
        auto.feed(Frame(img, i, T0 + timedelta(seconds=i / 15)))
    auto.wait(timeout=120)
    assert auto.calibration is not None
    assert (
        float(angular_separation_deg(auto.calibration.camera.boresight, truth.boresight))
        < 30 / 3600
    )
    assert cache.exists()
    # A restart picks up the cached solution immediately.
    assert AutoCalibrator(cfg, OBS, fps=15, cache_path=cache).calibration is not None
    auto.close()
