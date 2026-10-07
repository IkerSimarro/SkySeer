"""Phase 2: satellite identification against the real catalogue (frozen 2026-10-07 snapshot)."""

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load, wgs84

from skyseer.astro.calibrate import calibrate, load_sky_image
from skyseer.astro.frames import Observer, altaz_to_vec, vec_to_altaz
from skyseer.astro.orbits import OrbitCatalog, is_sunlit
from skyseer.astro.satellites import SatelliteIdentifier
from skyseer.astro.simulate import camera_looking_at, find_passes, render_sky_video
from skyseer.astro.sky import SkyPath
from skyseer.config import SatelliteConfig, load_config
from skyseer.pipeline import run
from skyseer.sources import FileSource

SNAPSHOT = Path(__file__).parent / "data" / "celestrak_2026-10-07.tle.gz"
MADRID = Observer(40.4168, -3.7038, 650)
EVENING = datetime(2026, 10, 7, 19, 50, tzinfo=UTC)


@pytest.fixture(scope="session")
def orbits() -> OrbitCatalog:
    return OrbitCatalog.from_tle_text(gzip.decompress(SNAPSHOT.read_bytes()).decode("ascii"))


def test_snapshot_loads_and_dedupes(orbits):
    assert len(orbits) > 15000
    assert len(set(orbits.norad.tolist())) == len(orbits)  # one (newest) set per object
    assert abs(float(np.median(orbits.epoch_age_days(EVENING)))) < 3


def test_tle_parsing_two_and_three_line_and_newest_wins():
    l1_old = "1 25544U 98067A   26270.50000000  .00016717  00000+0  10270-3 0  9001"
    l2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    l1_new = "1 25544U 98067A   26279.50000000  .00016717  00000+0  10270-3 0  9002"
    cat = OrbitCatalog.from_tle_text(f"ISS (ZARYA)\n{l1_old}\n{l2}\n{l1_new}\n{l2}\n")
    assert len(cat) == 1 and cat.norad[0] == 25544
    assert cat.epochs[0] == pytest.approx(2461320.0)  # day 279.5 of 2026 = 6 Oct 12:00 UTC
    assert cat.names[0] == "ISS (ZARYA)"  # name kept although the newest set had none


def test_propagation_agrees_with_skyfield(orbits):
    ts = load.timescale(builtin=True)
    topos = wgs84.latlon(MADRID.latitude, MADRID.longitude, elevation_m=MADRID.elevation_m)
    idx = np.random.default_rng(0).choice(len(orbits), 25, replace=False)
    times = [EVENING, EVENING + timedelta(seconds=37)]
    enu, rng_km, _, valid = orbits.subset(idx).topocentric(times, MADRID)
    for j, i in enumerate(idx):
        sat = EarthSatellite.from_satrec(orbits.satrecs[i], ts)
        for k, when in enumerate(times):
            if not valid[j, k]:
                continue
            alt, az, dist = (sat - topos).at(ts.from_datetime(when)).altaz()
            ref = altaz_to_vec(alt.degrees, az.degrees)
            assert np.degrees(np.arccos(np.clip(ref @ enu[j, k], -1, 1))) < 30 / 3600
            assert dist.km == pytest.approx(rng_km[j, k], rel=1e-4)


def test_earth_shadow():
    noon_sun_side = np.array([[7000.0, 0.0, 0.0]])
    when = datetime(2026, 3, 20, 12, tzinfo=UTC)  # equinox: Sun near +x (vernal equinox)
    assert is_sunlit(noon_sun_side, when)[0]
    assert not is_sunlit(-noon_sun_side, when)[0]  # directly behind the Earth
    assert is_sunlit(np.array([[-7000.0, 0.0, 9000.0]]), when)[0]  # behind, but far above


@pytest.fixture(scope="module")
def satellite_video(tmp_path_factory, orbits, catalog):
    cam = camera_looking_at(50, 250, 2, 1280, 720, 55, k1=-0.05)
    passes = find_passes(orbits, MADRID, cam, EVENING, 20, min_seconds_in_view=6)
    assert len(passes) >= 3, "scenario should contain several real satellites"
    path = tmp_path_factory.mktemp("sat") / "madrid.mp4"
    truth = render_sky_video(
        path, cam, MADRID, EVENING, 20, 10, catalog, orbits, [i for i, _ in passes[:5]]
    )
    return path, truth


def _identify_video(path, start, orbits, test_db, max_clock=3.0):
    cfg = load_config(satellites={"max_clock_offset_s": max_clock})
    image, when = load_sky_image(path, start, 0, 2.0)
    cal = calibrate(image, when, MADRID, fov_estimate=55, fov_max_error=10, database=test_db)
    ident = SatelliteIdentifier(orbits, MADRID, cfg.satellites)
    return list(run(FileSource(path, start), cfg, calibration=cal, identifier=ident))


def _event_for(truth_pass, events):
    tr = {f: (x, y) for f, x, y in truth_pass.points}

    def hits(e):
        return sum(
            1
            for d in e.track.detections
            if d.frame_index in tr
            and np.hypot(d.x - tr[d.frame_index][0], d.y - tr[d.frame_index][1]) < 5
        )

    best = max(events, key=hits, default=None)
    return best if best is not None and hits(best) > 0 else None


@pytest.mark.parametrize("clock_error_s", [0.0, 2.0])
def test_names_real_satellites_end_to_end(satellite_video, orbits, test_db, clock_error_s):
    """Real orbits drawn into a real star field go through the normal pipeline; every
    satellite must be named correctly, and a wrong camera clock must be measured."""
    path, truth = satellite_video
    events = _identify_video(path, EVENING + timedelta(seconds=clock_error_s), orbits, test_db)
    for p in truth:
        ev = _event_for(p, events)
        assert ev is not None, f"{p.name} not detected"
        assert ev.identification is not None, f"{p.name} not identified"
        assert ev.identification.norad_id == p.norad_id
        assert ev.label_source == "orbit"
        assert ev.identification.clock_offset_s == pytest.approx(-clock_error_s, abs=0.15)
        assert ev.identification.cross_track_deg < 0.03
        assert ev.identification.sunlit


def test_random_paths_are_not_named(orbits):
    """Satellite-like paths that are NOT catalogued objects must stay unidentified."""
    ident = SatelliteIdentifier(orbits, MADRID, SatelliteConfig())
    rng = np.random.default_rng(1)
    for _ in range(60):
        t0 = EVENING + timedelta(seconds=float(rng.uniform(-1800, 1800)))
        start = altaz_to_vec(rng.uniform(20, 80), rng.uniform(0, 360))
        axis = np.cross(start, altaz_to_vec(rng.uniform(0, 90), rng.uniform(0, 360)))
        axis /= np.linalg.norm(axis)
        speed = rng.uniform(0.3, 1.5)
        times = [t0 + timedelta(seconds=s) for s in np.arange(0, 10, 0.2)]
        vecs = np.array(
            [
                start * np.cos(np.radians(speed * s))
                + np.cross(axis, start) * np.sin(np.radians(speed * s))
                for s in np.arange(0, 10, 0.2)
            ]
        )
        alt, az = vec_to_altaz(vecs)
        path = SkyPath(times, np.zeros(len(times)), np.zeros(len(times)), alt, az)
        assert ident.identify(path) is None


def test_old_recordings_need_matching_orbits(orbits):
    """Orbits from October 2026 must not be used to 'identify' a January 2025 video."""
    ident = SatelliteIdentifier(orbits, MADRID, SatelliteConfig())
    t = [datetime(2025, 1, 31, 20, tzinfo=UTC) + timedelta(seconds=i) for i in range(5)]
    path = SkyPath(t, np.zeros(5), np.zeros(5), np.full(5, 45.0), np.linspace(100, 104, 5))
    assert not ident.can_check(path)
    assert ident.identify(path) is None
