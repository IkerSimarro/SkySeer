"""Phase 3: meteor showers, photometry, light curves and fireball alerts."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from skyseer.alerts import Alerter
from skyseer.astro.calibrate import calibrate, load_sky_image
from skyseer.astro.frames import (
    Observer,
    apply_refraction,
    icrs_to_enu_matrix,
    remove_refraction,
    vec_to_altaz,
    vec_to_radec,
)
from skyseer.astro.lightcurve import find_period
from skyseer.astro.photometry import EventPhotometry
from skyseer.astro.showers import ShowerAssociator, active_showers, load_showers, solar_longitude
from skyseer.astro.simulate import (
    camera_looking_at,
    flux_of_magnitude,
    render_meteor_video,
    shower_meteor,
    sporadic_meteor,
)
from skyseer.astro.sky import SkyPath
from skyseer.config import AlertConfig, PhotometryConfig, ShowerConfig, load_config
from skyseer.models import Event, ObjectClass, Track
from skyseer.pipeline import run
from skyseer.sources import FileSource

from .conftest import make_detection

MADRID = Observer(40.4168, -3.7038, 650)
PERSEID_NIGHT = datetime(2026, 8, 12, 23, 0, tzinfo=UTC)


def _shower(code: str):
    return next(s for s in load_showers() if s.code == code)


# ----------------------------------------------------------------------------- dates & data
@pytest.mark.parametrize(
    ("when", "expected"),
    [  # 2026 equinoxes/solstices: 0/90/180/270 of date = ~0.36 deg less in J2000
        (datetime(2026, 3, 20, 14, 46, tzinfo=UTC), 359.64),
        (datetime(2026, 6, 21, 8, 24, tzinfo=UTC), 89.64),
        (datetime(2026, 9, 23, 0, 5, tzinfo=UTC), 179.64),
        (datetime(2026, 12, 21, 20, 50, tzinfo=UTC), 269.64),
    ],
)
def test_solar_longitude(when, expected):
    assert solar_longitude(when) == pytest.approx(expected, abs=0.05)


def test_shower_table_matches_published_values():
    """Spot-check the bundled IAU MDC data against the well-known major showers."""
    per, gem, ori = _shower("PER"), _shower("GEM"), _shower("ORI")
    assert (per.ra, per.dec, per.vg) == pytest.approx((48, 58, 59), abs=1.5)
    assert (gem.ra, gem.dec, gem.vg) == pytest.approx((113, 32, 34), abs=1.5)
    assert (ori.ra, ori.dec, ori.vg) == pytest.approx((96, 16, 66), abs=1.5)
    assert per.lsun_peak == pytest.approx(140, abs=0.5)
    assert len(load_showers()) > 100
    # Perseid radiant drifts ~1.4 deg/day in RA: 10 days before the peak it is ~14 deg west.
    assert per.radiant(per.lsun_peak - 10)[0] == pytest.approx(per.ra - 14.5, abs=1.0)
    assert "ORI" in {s.code for s in active_showers(datetime(2026, 10, 21, 22, tzinfo=UTC))}
    assert "PER" not in {s.code for s in active_showers(datetime(2026, 3, 1, tzinfo=UTC))}


# ----------------------------------------------------------------------------- association
def _path(meteor, rng, noise_deg=0.03) -> SkyPath:
    ts = np.arange(0, meteor.duration, 0.04)
    v = apply_refraction(np.array([meteor.position(t) for t in ts]))
    v += rng.normal(0, np.radians(noise_deg), v.shape)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    times = [meteor.start_time + timedelta(seconds=float(t)) for t in ts]
    alt, az = vec_to_altaz(v)
    ra, dec = vec_to_radec(remove_refraction(v) @ icrs_to_enu_matrix(meteor.start_time, MADRID))
    return SkyPath(times, ra, dec, alt, az)


def test_perseid_is_recognised_and_reversed_path_is_not():
    rng = np.random.default_rng(1)
    cam = camera_looking_at(55, 115, 0, 1280, 720, 55)
    m = shower_meteor(_shower("PER"), PERSEID_NIGHT, MADRID, cam, rng)
    assert m is not None
    match = ShowerAssociator(ShowerConfig(), MADRID).associate(_path(m, rng))
    assert match is not None and match.code == "PER"
    assert match.radiant_offset_deg < 0.5 and 0.85 < match.speed_ratio < 1.15
    # The same path flown towards the radiant cannot come from it.
    m.start, m.end = m.end, m.start
    rev = ShowerAssociator(ShowerConfig(), MADRID).associate(_path(m, rng))
    assert rev is None or rev.code != "PER"


def test_association_statistics():
    """Realistic scatter: most shower meteors found, few sporadics matched by chance."""
    rng = np.random.default_rng(3)
    assoc = ShowerAssociator(ShowerConfig(), MADRID)
    hits = total = false = sporadic = 0
    for i in range(600):
        when = datetime(2026, 1, 1, 22, tzinfo=UTC) + timedelta(
            days=float(rng.uniform(0, 365)), hours=float(rng.uniform(0, 6))
        )
        cam = camera_looking_at(rng.uniform(35, 80), rng.uniform(0, 360), 0, 1280, 720, 55)
        if i % 2 == 0:
            candidates = active_showers(when, 1000)
            if not candidates:
                continue
            sh = candidates[rng.integers(len(candidates))]
            m = shower_meteor(sh, when, MADRID, cam, rng, radiant_spread_deg=1.0, speed_spread=0.15)
            if m is None:
                continue
            match = assoc.associate(_path(m, rng))
            total += 1
            hits += int(match is not None and match.code == sh.code)
        else:
            sporadic += 1
            false += int(assoc.associate(_path(sporadic_meteor(when, cam, rng), rng)) is not None)
    assert hits / total > 0.9
    assert false / sporadic < 0.08


@pytest.fixture(scope="module")
def perseid_video(tmp_path_factory, catalog):
    rng = np.random.default_rng(7)
    cam = camera_looking_at(55, 115, 0, 1280, 720, 55, k1=-0.05)
    per = _shower("PER")
    meteors = []
    for k, mag in enumerate([1.6, 2.0, 1.8, 2.2]):
        when = PERSEID_NIGHT + timedelta(seconds=3 + 3 * k)
        meteors.append(
            shower_meteor(per, when, MADRID, cam, rng, mag)
            if k != 2
            else sporadic_meteor(when, cam, rng, mag)
        )
    path = tmp_path_factory.mktemp("meteors") / "perseids.mp4"
    render_meteor_video(path, cam, MADRID, PERSEID_NIGHT, 16, 25, catalog, meteors)
    return path, meteors


def test_perseid_night_end_to_end(perseid_video, test_db):
    """Simulated Perseids over Madrid go through the normal pipeline: every meteor must
    be detected, given the right shower (or sporadic) and a magnitude within 0.4."""
    path, meteors = perseid_video
    image, when = load_sky_image(path, PERSEID_NIGHT, 0, 2.0)
    cal = calibrate(image, when, MADRID, fov_estimate=55, fov_max_error=10, database=test_db)
    # Expected zero point for the simulator's flux scale (99% of a star's light falls in
    # the 4 px aperture).
    expected_zp = 2.0 + 2.5 * np.log10(flux_of_magnitude(2.0) * 0.99)
    assert cal.zero_point is not None
    assert cal.zero_point.value == pytest.approx(expected_zp, abs=0.25)

    events = list(
        run(
            FileSource(path, PERSEID_NIGHT), load_config(alerts={"log_file": None}), calibration=cal
        )
    )
    for m in meteors:
        pts = {f: (x, y) for f, x, y in m.points}

        def hits(e, pts=pts):
            return sum(
                1
                for d in e.track.detections
                if d.frame_index in pts
                and np.hypot(d.x - pts[d.frame_index][0], d.y - pts[d.frame_index][1]) < 25
            )

        ev = max(events, key=hits)
        assert hits(ev) > 3, f"meteor at {m.start_time} not detected"
        assert ev.label is ObjectClass.METEOR
        assert ev.shower_checked
        assert (ev.shower.code if ev.shower else None) == m.shower
        assert ev.photometry.peak_magnitude == pytest.approx(m.peak_mag, abs=0.4)


# ----------------------------------------------------------------------------- light curves
def test_tumbling_period_is_found():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 600))  # uneven sampling, like a track with gaps
    mag = 5 + 0.6 * np.sin(2 * np.pi * t / 2.7) + rng.normal(0, 0.15, t.size)
    p = find_period(t, mag)
    assert p is not None and p.period_s == pytest.approx(2.7, rel=0.02)
    assert p.false_alarm < 1e-6
    assert find_period(t, 5 + rng.normal(0, 0.3, t.size)) is None  # pure noise


# ----------------------------------------------------------------------------- alerts
def _meteor_event(peak_mag: float) -> Event:
    track = Track(1, 30.0, [make_detection(i, 100 + 10 * i, 50) for i in range(10)])
    ev = Event(1, track, ObjectClass.METEOR, 0.9, {}, {})
    ev.photometry = EventPhotometry(peak_mag, False, [peak_mag] * 10)
    return ev


def test_fireball_alert(tmp_path):
    log_file = tmp_path / "alerts.jsonl"
    alerter = Alerter(AlertConfig(log_file=str(log_file)), PhotometryConfig())
    assert not alerter.check(_meteor_event(-1.0))  # bright, but not a fireball
    assert alerter.check(_meteor_event(-5.2))
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["peak_magnitude"] == -5.2
    assert "Fireball" in alerter.sent[0]["message"]
