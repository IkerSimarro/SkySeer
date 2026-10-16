"""Sun altitude and twilight, checked against Skyfield with NASA JPL's DE421 ephemeris."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from skyseer.astro.frames import Observer
from skyseer.astro.sun import dark_windows, is_dark, next_change, sun_altitude

MADRID = Observer(40.42, -3.70, 650)
DE421 = Path("data/astro/de421.bsp")


@pytest.fixture(scope="module")
def skyfield_sun():
    if not DE421.exists():
        pytest.skip("JPL ephemeris not downloaded")
    from skyfield.api import Loader, wgs84

    load = Loader(str(DE421.parent))
    eph = load("de421.bsp")
    ts = load.timescale()

    def altitude(when: datetime, obs: Observer) -> float:
        place = eph["earth"] + wgs84.latlon(obs.latitude, obs.longitude, obs.elevation_m)
        alt, _, _ = place.at(ts.from_datetime(when)).observe(eph["sun"]).apparent().altaz()
        return float(alt.degrees)

    return altitude


def test_sun_altitude_matches_jpl(skyfield_sun):
    rng = np.random.default_rng(0)
    places = [MADRID, Observer(-33.9, 151.2), Observer(64.1, -21.9), Observer(0.0, 0.0)]
    worst = 0.0
    for _ in range(40):
        when = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=float(rng.uniform(0, 8760)))
        obs = places[int(rng.integers(len(places)))]
        worst = max(worst, abs(sun_altitude(when, obs) - skyfield_sun(when, obs)))
    assert worst < 0.05  # degrees; twilight needs ~0.5


def test_nautical_dusk_time_matches_jpl(skyfield_sun):
    noon = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    dusk = next_change(noon, MADRID, -12.0)
    # JPL (Skyfield almanac): nautical twilight ends at 18:25:36 UTC that evening.
    jpl = noon.replace(hour=18, minute=25, second=36)
    assert dusk is not None and abs((dusk - jpl).total_seconds()) < 10
    # At the moment found, JPL puts the Sun at -12° too (within a few seconds of motion).
    assert abs(skyfield_sun(dusk, MADRID) - (-12.0)) < 0.1


def test_dark_windows_over_a_week():
    start = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    windows = dark_windows(start, start + timedelta(days=7), MADRID, -12.0)
    assert len(windows) == 7
    for begin, end in windows:
        hours = (end - begin).total_seconds() / 3600
        assert 10.5 < hours < 11.5  # nautical night in Madrid in mid-October
        assert not is_dark(begin - timedelta(minutes=1), MADRID)
        assert is_dark(begin + timedelta(minutes=1), MADRID)


def test_polar_summer_has_no_night():
    tromso = Observer(69.65, 18.96)
    assert next_change(datetime(2026, 6, 21, tzinfo=UTC), tromso, -12.0) is None
