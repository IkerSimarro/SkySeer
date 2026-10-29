"""The Moon's position and phase, checked against Skyfield with NASA JPL's DE421 ephemeris."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from skyseer.astro.frames import Observer, altaz_to_vec, angular_separation_deg
from skyseer.astro.moon import (
    illuminated_fraction,
    moon_enu,
    moon_near_path,
    moon_over_night,
    moon_state,
)
from skyseer.astro.simulate import camera_looking_at, render_moon
from skyseer.astro.sky import SkyPath

MADRID = Observer(40.42, -3.70, 650)
DE421 = Path("data/astro/de421.bsp")


@pytest.fixture(scope="module")
def jpl():
    if not DE421.exists():
        pytest.skip("JPL ephemeris not downloaded")
    from skyfield.api import Loader, wgs84

    load = Loader(str(DE421.parent))
    eph = load("de421.bsp")
    ts = load.timescale()

    def moon(when: datetime, obs: Observer):
        t = ts.from_datetime(when)
        place = eph["earth"] + wgs84.latlon(obs.latitude, obs.longitude, obs.elevation_m)
        alt, az, dist = place.at(t).observe(eph["moon"]).apparent().altaz()
        return altaz_to_vec(alt.degrees, az.degrees), dist.km

    return moon, eph, ts


def test_moon_position_matches_jpl(jpl):
    moon, _, _ = jpl
    rng = np.random.default_rng(0)
    places = [MADRID, Observer(-33.9, 151.2), Observer(64.1, -21.9), Observer(0.0, 0.0)]
    worst_deg = worst_km = 0.0
    for _ in range(60):
        when = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=float(rng.uniform(0, 8760)))
        obs = places[int(rng.integers(len(places)))]
        ref, ref_km = moon(when, obs)
        v, km = moon_enu(when, obs, refraction=False)
        worst_deg = max(worst_deg, float(angular_separation_deg(v, ref)))
        worst_km = max(worst_km, abs(km - ref_km))
    # Seen from the Earth's surface (parallax included); a Pi camera pixel is ~0.03°.
    assert worst_deg < 0.01
    assert worst_km < 100


def test_fraction_lit_matches_jpl(jpl):
    _, eph, ts = jpl
    from skyfield import almanac

    for days in range(0, 30, 3):
        when = datetime(2026, 12, 1, tzinfo=UTC) + timedelta(days=days)
        ref = almanac.fraction_illuminated(eph, "moon", ts.from_datetime(when))
        assert illuminated_fraction(when) == pytest.approx(ref, abs=0.001)


def test_phases_of_december_2026():
    # JPL (Skyfield almanac): new Moon 9 Dec 00:52 UTC, full Moon 24 Dec 01:28 UTC.
    new = moon_state(datetime(2026, 12, 9, 0, 52, tzinfo=UTC), MADRID)
    full = moon_state(datetime(2026, 12, 24, 1, 28, tzinfo=UTC), MADRID)
    assert new.fraction < 0.01 and new.phase == "new Moon"
    assert full.fraction > 0.99 and full.phase == "full Moon"
    geminids = moon_state(datetime(2026, 12, 14, 0, 0, tzinfo=UTC), MADRID)
    assert geminids.phase == "waxing crescent" and not geminids.up  # set by 20:50 UTC
    assert 0.24 < geminids.radius_deg < 0.28  # half a degree across


def test_the_moon_over_a_night():
    # The Geminid peak night from Madrid: a thin crescent, up only in the early evening.
    night = moon_over_night(
        datetime(2026, 12, 13, 18, 0, tzinfo=UTC), datetime(2026, 12, 14, 6, 0, tzinfo=UTC), MADRID
    )
    assert night["phase"] == "waxing crescent"
    assert 0.15 < night["fraction_lit"] < 0.3
    assert 2.0 < night["hours_up"] < 4.0  # sets at 20:50 UTC


FULL_MOON_UP = datetime(2026, 10, 23, 19, 0, tzinfo=UTC)  # 93% lit, alt 33.5°, az 121°


def _path_through(alt, az, when):
    times = [when + timedelta(seconds=k) for k in range(5)]
    alt = np.array([alt] * 5)
    az = np.array([az - 2.0, az - 1.0, az, az + 1.0, az + 2.0])
    return SkyPath(times, np.zeros(5), np.zeros(5), alt, az)


def test_paths_near_the_moon_are_marked():
    moon = moon_state(FULL_MOON_UP, MADRID)
    near = moon_near_path(_path_through(moon.alt + 1.0, moon.az, FULL_MOON_UP), MADRID, 5.0)
    assert near is not None and near["separation_deg"] == pytest.approx(1.0, abs=0.05)
    assert near["fraction_lit"] == pytest.approx(0.93, abs=0.01)
    far = _path_through(moon.alt + 20.0, moon.az, FULL_MOON_UP)
    assert moon_near_path(far, MADRID, 5.0) is None
    # Twelve hours later the Moon is below the horizon: nothing is near it.
    later = FULL_MOON_UP + timedelta(hours=12)
    assert not moon_state(later, MADRID).up
    assert moon_near_path(_path_through(moon.alt, moon.az, later), MADRID, 5.0) is None


def test_minutes_in_view_follow_the_camera():
    moon = moon_state(FULL_MOON_UP, MADRID)
    end = FULL_MOON_UP + timedelta(minutes=30)
    at_it = camera_looking_at(moon.alt, moon.az, 0, 1280, 720, 55)
    away = camera_looking_at(60, (moon.az + 180) % 360, 0, 1280, 720, 55)
    assert moon_over_night(FULL_MOON_UP, end, MADRID, at_it)["minutes_in_view"] == 30.0
    assert moon_over_night(FULL_MOON_UP, end, MADRID, away)["minutes_in_view"] == 0.0
    one_minute = moon_over_night(FULL_MOON_UP, FULL_MOON_UP + timedelta(minutes=1), MADRID)
    assert one_minute["hours_up"] == pytest.approx(1 / 60, abs=0.001)


def test_the_simulated_moon_is_where_the_moon_is():
    moon = moon_state(FULL_MOON_UP, MADRID)
    cam = camera_looking_at(moon.alt + 10, moon.az + 10, 0, 1280, 720, 55)
    layer, glow = render_moon(cam, MADRID, FULL_MOON_UP)
    uv, _ = cam.vec_to_pixel(moon.enu[None, :])
    x, y = (int(round(float(c))) for c in uv[0])
    assert layer[y, x] >= 255  # the disc is saturated
    assert glow > 0 and layer[5, 5] < 60  # far corner: only moonlit sky
    gx, gy = int(2 * cam.cx - x), int(2 * cam.cy - y)  # the lens ghost, mirrored
    assert layer[gy, gx] > glow + 10


def test_the_night_report_tells_about_the_moon():
    from skyseer.reports import night_report

    night = moon_over_night(FULL_MOON_UP, FULL_MOON_UP + timedelta(hours=1), MADRID)
    session = {"id": 1, "name": "n", "started": FULL_MOON_UP.isoformat(), "meta": {"moon": night}}
    base = {"label": "Satellite", "start_time": FULL_MOON_UP.isoformat()}
    base["end_time"] = (FULL_MOON_UP + timedelta(seconds=30)).isoformat()
    near = base | {"id": 1, "moon": {"separation_deg": 1.5, "fraction_lit": 0.93}}
    report = night_report(session, [near, base | {"id": 2}])
    assert report["moon"]["phase"] == "waxing gibbous" and report["near_moon"] == 1


def test_the_live_station_reports_the_moon(synth_clip, tmp_path):
    from skyseer.config import load_config
    from skyseer.live import LiveStation
    from skyseer.sources import ReplaySource
    from skyseer.storage import Store

    path, _ = synth_clip
    source = ReplaySource(path, start_time=FULL_MOON_UP, speed=0.0)
    store = Store(tmp_path / "db.sqlite")
    station = LiveStation(source, load_config(), store, "replay", tmp_path / "n", MADRID)
    moon = station.status()["moon"]
    assert moon["phase"] == "waxing gibbous" and moon["up"]
    assert "in_view" not in moon  # not calibrated yet: where it is in the picture is unknown
    without = LiveStation(ReplaySource(path, speed=0.0), load_config(), store, "replay", tmp_path)
    assert without.status()["moon"] is None  # no location, no Moon
