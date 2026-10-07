from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from skyseer.astro.calibrate import Calibration, CalibrationError, calibrate
from skyseer.astro.camera import CameraModel, fit_camera, wahba_rotation
from skyseer.astro.frames import (
    Observer,
    altaz_to_vec,
    angular_separation_deg,
    apply_refraction,
    icrs_to_enu_matrix,
    radec_to_vec,
    refraction_deg,
    remove_refraction,
    vec_to_altaz,
)
from skyseer.astro.simulate import camera_looking_at, render_starfield
from skyseer.astro.starnames import STAR_NAMES


# --------------------------------------------------------------------------- frames
@pytest.mark.parametrize("lat", [-30.0, 10.0, 40.0, 65.0])
def test_celestial_pole_altitude_equals_latitude(lat):
    """The sky pole sits at an altitude equal to the observer's latitude, due N or S."""
    obs = Observer(lat, 12.0)
    pole = radec_to_vec(0.0, 90.0 if lat > 0 else -90.0)
    for hours in (0, 6, 13):
        when = datetime(2026, 3, 1, tzinfo=UTC) + timedelta(hours=hours)
        alt, az = vec_to_altaz(pole @ icrs_to_enu_matrix(when, obs).T)
        # Small offset: the catalogue pole is J2000, the real pole has precessed ~0.15°.
        assert float(alt) == pytest.approx(abs(lat), abs=0.3)
        assert min(float(az), 360 - float(az)) < 1.0 or abs(float(az) - 180) < 1.0


def test_esa_real_images_alt_az():
    """ESA tetra3's sample photos (ESTEC, 2019-07-29T20:47:26Z) are named with the mount's
    nominal pointing: Alt40/Azi-135 and Alt60/Azi-135. Our plate-solved centres, converted
    to the horizon frame, must land within a degree or so of those settings."""
    obs = Observer(52.2188, 4.4206)
    when = datetime(2019, 7, 29, 20, 47, 26, tzinfo=UTC)
    for (ra, dec), nominal_alt in (((230.668, 11.035), 40), ((240.464, 28.94), 60)):
        alt, az = vec_to_altaz(
            apply_refraction(radec_to_vec(ra, dec) @ icrs_to_enu_matrix(when, obs).T)
        )
        assert abs(float(alt) - nominal_alt) < 1.5
        assert abs(float(az) - 225.0) < 1.5


def test_refraction_magnitude_and_inverse():
    assert refraction_deg(0.0) == pytest.approx(0.48, abs=0.05)  # ~29' at the horizon
    assert refraction_deg(90.0) < 0.001
    v = altaz_to_vec(np.array([5.0, 20.0, 60.0]), np.array([10.0, 100.0, 250.0]))
    back = remove_refraction(apply_refraction(v))
    assert np.all(angular_separation_deg(v, back) < 1e-4)


# --------------------------------------------------------------------------- camera
def test_camera_roundtrip_with_distortion():
    cam = camera_looking_at(45, 120, 7, 1920, 1080, 55, k1=-0.08, k2=0.01)
    uv = np.random.default_rng(0).uniform([0, 0], [1920, 1080], size=(200, 2))
    back, front = cam.vec_to_pixel(cam.pixel_to_vec(uv))
    assert front.all()
    assert np.abs(back - uv).max() < 1e-6


def test_directions_outside_view_never_fold_into_image():
    """With barrel distortion the polynomial bends back far off-axis; such directions must
    be reported as not visible rather than landing inside the frame."""
    cam = camera_looking_at(60, 45, 0, 1920, 1080, 55, k1=-0.06, k2=-0.009)
    az = np.linspace(0, 360, 721)
    v = altaz_to_vec(np.full_like(az, 20.0), az)  # a ring 40° below the boresight
    uv, ok = cam.vec_to_pixel(v)
    angle = angular_separation_deg(v, cam.boresight)
    assert not cam.in_image(uv, ok)[angle > 40].any()


def test_pointing_reports_alt_az_roll():
    p = camera_looking_at(35, 250, -8, 1920, 1080, 55).pointing_altaz()
    assert p["alt"] == pytest.approx(35, abs=1e-6)
    assert p["az"] == pytest.approx(250, abs=1e-6)
    assert p["roll"] == pytest.approx(-8, abs=1e-6)


def test_wahba_recovers_rotation():
    from scipy.spatial.transform import Rotation

    rot = Rotation.random(random_state=1).as_matrix()
    v = Rotation.random(50, random_state=2).apply([0, 0, 1])
    assert np.allclose(wahba_rotation(v, v @ rot.T), rot, atol=1e-9)


def test_fit_camera_recovers_lens():
    truth = camera_looking_at(50, 30, 3, 1920, 1080, 55, k1=-0.07)
    uv = np.random.default_rng(3).uniform([0, 0], [1920, 1080], size=(80, 2))
    fit = fit_camera(uv, truth.pixel_to_vec(uv), 1920, 1080, f_init=1700, fit_center=True)
    assert fit.rms_px < 0.01
    assert fit.model.k1 == pytest.approx(-0.07, abs=1e-3)
    assert fit.model.f == pytest.approx(truth.f, rel=1e-4)


def test_folding_distortion_is_unphysical():
    assert camera_looking_at(45, 0, 0, 1920, 1080, 55, k1=-0.08).distortion_is_physical()
    # The model fitted to a false match in a real screenshot (99° FOV): it folds back on
    # itself before the image corner, which no real lens does.
    assert not CameraModel(1920, 1080, 820, 960, 540, k1=-0.525, k2=-0.343).distortion_is_physical()


# --------------------------------------------------------------------------- calibration
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrates_simulated_sky(seed, catalog, test_db):
    rng = np.random.default_rng(100 + seed)
    obs = Observer(rng.uniform(-40, 60), rng.uniform(-180, 180), 300)
    when = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=float(rng.uniform(0, 8760)))
    truth = camera_looking_at(
        rng.uniform(30, 80),
        rng.uniform(0, 360),
        rng.uniform(-10, 10),
        1920,
        1080,
        55.0,
        k1=float(rng.uniform(-0.1, -0.01)),
    )
    image = render_starfield(truth, obs, when, catalog, seed=seed)
    cal = calibrate(
        image, when, obs, fov_estimate=55, fov_max_error=10, database=test_db, catalog=catalog
    )

    assert float(angular_separation_deg(cal.camera.boresight, truth.boresight)) < 30 / 3600
    assert cal.camera.pointing_altaz()["roll"] == pytest.approx(
        truth.pointing_altaz()["roll"], abs=0.02
    )
    gx, gy = np.meshgrid(np.linspace(0, 1919, 20), np.linspace(0, 1079, 12))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    err = angular_separation_deg(cal.camera.pixel_to_vec(grid), truth.pixel_to_vec(grid))
    assert err.max() < truth.pixel_scale_arcsec * 1.5 / 3600  # < 1.5 px anywhere in frame


def test_calibration_save_load_roundtrip(tmp_path, catalog, test_db):
    obs = Observer(40.4, -3.7, 650)
    when = datetime(2026, 8, 12, 22, 30, tzinfo=UTC)
    truth = camera_looking_at(60, 45, 0, 1920, 1080, 55, k1=-0.05)
    cal = calibrate(
        render_starfield(truth, obs, when, catalog),
        when,
        obs,
        fov_estimate=55,
        fov_max_error=10,
        database=test_db,
        catalog=catalog,
    )
    cal.save(tmp_path / "cal.json")
    loaded = Calibration.load(tmp_path / "cal.json")
    uv = np.array([[100.0, 200.0], [1800.0, 900.0]])
    assert np.allclose(loaded.pixel_to_altaz(uv), cal.pixel_to_altaz(uv))
    ra, dec = loaded.pixel_to_radec(uv)
    back, _ = loaded.radec_to_pixel(ra, dec)
    assert np.abs(back - uv).max() < 0.01


def test_random_dots_are_never_accepted_as_a_sky(test_db, catalog):
    """A fake 'sky' of random dots must be rejected, not given a confident wrong answer."""
    from skyseer.synth import _stamp

    rng = np.random.default_rng(7)
    img = np.full((1080, 1920), 20.0, np.float32)
    for x, y, a in zip(
        rng.uniform(0, 1920, 150),
        rng.uniform(0, 1080, 150),
        rng.pareto(2.0, 150) * 40 + 20,
        strict=True,
    ):
        _stamp(img, x, y, min(a, 230), 1.3)
    img += rng.normal(0, 2.5, img.shape).astype(np.float32)
    with pytest.raises(CalibrationError):
        calibrate(
            np.clip(img, 0, 255).astype(np.uint8),
            datetime(2026, 1, 1, tzinfo=UTC),
            Observer(40, 0),
            fov_estimate=55,
            fov_max_error=10,
            database=test_db,
            catalog=catalog,
        )


def test_star_names_point_at_bright_stars(catalog):
    idx = {int(h): i for i, h in enumerate(catalog.hip)}
    for hip, name in STAR_NAMES.items():
        if name == "Mira":  # long-period variable; catalogue lists its faint mean magnitude
            continue
        assert hip in idx, name
        assert catalog.mag[idx[hip]] < 4.0, name
    vega = idx[91262]
    assert catalog.ra[vega] == pytest.approx(279.23, abs=0.05)
    assert catalog.dec[vega] == pytest.approx(38.78, abs=0.05)
