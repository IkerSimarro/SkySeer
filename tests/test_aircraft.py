"""Aircraft identification from ADS-B broadcasts: geometry, broadcasts, log, matching."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from skyseer.aircraft import (
    AircraftFeed,
    AircraftIdentifier,
    AircraftLog,
    AircraftState,
    RateLimited,
    apply_match,
    ecef_to_geodetic,
    enu_basis,
    geodetic_to_ecef,
    parse_opensky,
    parse_readsb,
    trajectory,
)
from skyseer.astro.frames import (
    Observer,
    altaz_to_vec,
    apply_refraction,
    remove_refraction,
    vec_to_altaz,
)
from skyseer.astro.simulate import flight_through
from skyseer.astro.sky import SkyPath
from skyseer.config import load_config
from skyseer.models import ObjectClass

MADRID = Observer(40.42, -3.70, 650)
EVENING = datetime(2026, 10, 19, 20, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- geometry
def test_directions_agree_with_skyfield():
    """Where an aircraft appears from the camera (alt/az and distance), checked against
    Skyfield's independent WGS84 implementation."""
    from skyfield.api import wgs84

    from skyseer.astro.frames import timescale

    t = timescale().utc(2026, 10, 19, 20)
    here = wgs84.latlon(MADRID.latitude, MADRID.longitude, elevation_m=MADRID.elevation_m)
    origin = geodetic_to_ecef(MADRID.latitude, MADRID.longitude, MADRID.elevation_m)
    rng = np.random.default_rng(0)
    for _ in range(50):
        lat = MADRID.latitude + rng.uniform(-1.2, 1.2)
        lon = MADRID.longitude + rng.uniform(-1.5, 1.5)
        h = rng.uniform(1000, 12500)
        alt, az, dist = (wgs84.latlon(lat, lon, elevation_m=h) - here).at(t).altaz()
        enu = (geodetic_to_ecef(lat, lon, h) - origin) @ enu_basis(
            MADRID.latitude, MADRID.longitude
        ).T
        my_alt, my_az = vec_to_altaz(enu / np.linalg.norm(enu))
        assert my_alt == pytest.approx(alt.degrees, abs=1 / 3600)  # within 1 arcsecond
        assert (my_az - az.degrees + 180) % 360 - 180 == pytest.approx(0, abs=1 / 3600)
        assert np.linalg.norm(enu) == pytest.approx(dist.m, abs=0.5)


def test_geodetic_round_trip():
    rng = np.random.default_rng(1)
    lat, lon, h = rng.uniform(-80, 80, 200), rng.uniform(-180, 180, 200), rng.uniform(0, 13e3, 200)
    back = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert np.abs(back[0] - lat).max() < 1e-8 and np.abs(back[2] - h).max() < 1e-3


# --------------------------------------------------------------------------- broadcasts
def test_opensky_and_readsb_answers_are_read():
    opensky = {
        "time": 1760900000,
        "states": [
            # icao24, callsign, country, time_position, last_contact, lon, lat, baro_alt,
            # on_ground, velocity, true_track, vertical_rate, sensors, geo_alt, squawk...
            ["3423c5", "IBE3456 ", "Spain", 1760899998, 1760899999, -3.5, 40.6, 10350.0,
             False, 231.5, 72.0, -0.3, None, 10500.0, "1000", False, 0],
            ["4ca123", "RYR12AB ", "Ireland", None, 1760899990, -3.9, 40.3, 9100.0,
             False, 220.0, 250.0, 5.0, None, None, None, False, 0],
            ["abc123", "GRND1", "Spain", 1760899990, 1760899990, -3.6, 40.5, None,
             True, 5.0, 10.0, 0.0, None, None, None, False, 0],
        ],
    }  # fmt: skip
    a, b = parse_opensky(opensky)  # the aircraft on the ground is dropped
    assert (a.icao24, a.callsign, a.time, a.alt_m, a.geometric) == (
        "3423c5", "IBE3456", 1760899998.0, 10500.0, True
    )  # fmt: skip
    assert a.country == "Spain" and a.speed_m_s == 231.5 and a.track_deg == 72.0
    assert (b.time, b.alt_m, b.geometric) == (1760899990.0, 9100.0, False)  # barometric

    readsb = {
        "now": 1760900000.5,
        "aircraft": [
            {"hex": "3423c5", "flight": "IBE3456 ", "lat": 40.6, "lon": -3.5,
             "alt_baro": 34000, "alt_geom": 34450, "gs": 450.0, "track": 72.0,
             "geom_rate": -64, "seen_pos": 0.5},
            {"hex": "4ca123", "alt_baro": 30000},  # no position yet
            {"hex": "abc123", "lat": 40.5, "lon": -3.6, "alt_baro": "ground"},
        ],
    }  # fmt: skip
    (c,) = parse_readsb(readsb)
    assert c.time == 1760900000.0 and c.alt_m == pytest.approx(34450 * 0.3048)
    assert c.speed_m_s == pytest.approx(450 * 1852 / 3600)  # knots
    assert c.vertical_m_s == pytest.approx(-64 * 0.3048 / 60)  # feet per minute


# --------------------------------------------------------------------------- log
def _state(icao, t, **kw):
    base = {"lat": 40.5, "lon": -3.6, "alt_m": 10000.0}
    return AircraftState(icao, t, **(base | kw))


def test_log_keeps_order_drops_repeats_and_saves(tmp_path):
    path = tmp_path / "aircraft.jsonl"
    log = AircraftLog(path=path)
    assert log.add([_state("a", 20), _state("a", 10), _state("b", 15)]) == 3
    assert log.add([_state("a", 10), _state("a", 30)]) == 1  # the repeat is ignored
    assert [s.time for s in log.states("a")] == [10, 20, 30]
    assert log.span() == (10, 30) and sorted(log.ids_between(25, 40)) == ["a"]
    again = AircraftLog.load(path)
    assert [s.time for s in again.states("a")] == [10, 20, 30] and len(again) == 4
    short = AircraftLog(keep_s=12)
    short.add([_state("a", 0), _state("b", 15), _state("b", 20)])
    assert short.states("a") == [] and len(short) == 2  # older than 12 s: forgotten


def test_trajectory_interpolates_and_dead_reckons():
    east_at = SimpleNamespace(speed_m_s=200.0, track_deg=90.0, vertical_m_s=0.0)
    a = _state("a", 0, **vars(east_at))
    b = _state("a", 10, lon=-3.6 + 2000 / (111_320 * np.cos(np.radians(40.5))), **vars(east_at))
    xyz, ok = trajectory([a, b], np.array([5.0, 15.0, 40.0]), max_gap_s=30, max_extrapolate_s=15)
    assert ok.tolist() == [True, True, False]  # 40 s is beyond dead-reckoning range
    pa, pb = geodetic_to_ecef(a.lat, a.lon, a.alt_m), geodetic_to_ecef(b.lat, b.lon, b.alt_m)
    assert np.linalg.norm(xyz[0] - (pa + pb) / 2) < 0.01  # half way at 5 s
    assert np.linalg.norm(xyz[1] - pb) == pytest.approx(1000, rel=0.01)  # 5 s x 200 m/s


# --------------------------------------------------------------------------- matching
def _flight(direction_altaz, heading, icao, callsign, altitude=10_500.0, speed=230.0):
    return flight_through(
        MADRID, altaz_to_vec(*direction_altaz), EVENING, altitude, speed, heading, icao, callsign
    )


def _observe(flight, seconds=40.0, clock_offset_s=0.0, noise_deg=0.02, seed=0):
    """The sky path a calibrated camera would measure (camera clock off by
    ``clock_offset_s``: true time = camera time + offset)."""
    rng = np.random.default_rng(seed)
    true_t = EVENING.timestamp() - seconds / 2 + np.arange(0, seconds, 0.2)
    origin = geodetic_to_ecef(MADRID.latitude, MADRID.longitude, MADRID.elevation_m)
    enu = (flight.position(true_t) - origin) @ enu_basis(MADRID.latitude, MADRID.longitude).T
    v = apply_refraction(enu / np.linalg.norm(enu, axis=1)[:, None])
    v = v + rng.normal(0, np.radians(noise_deg), v.shape)
    alt, az = vec_to_altaz(v / np.linalg.norm(v, axis=1)[:, None])
    times = [
        datetime.fromtimestamp(t - clock_offset_s, UTC) for t in true_t
    ]  # what the camera's clock said
    return SkyPath(times, np.zeros(len(times)), np.zeros(len(times)), alt, az)


def _log_of(flights, every_s=10.0, seed=0):
    rng = np.random.default_rng(seed)
    log = AircraftLog()
    for f in flights:
        log.add(f.broadcasts(EVENING - timedelta(seconds=90), 180, every_s, 20.0, rng))
    return log


def _decoys(n, seed):
    rng = np.random.default_rng(seed)
    return [
        _flight(
            (rng.uniform(15, 80), rng.uniform(0, 360)),
            rng.uniform(0, 360),
            f"d{k:05x}",
            f"DEC{k}",
            altitude=rng.uniform(3000, 12000),
        )
        for k in range(n)
    ]


@pytest.mark.parametrize("every_s", [5.0, 10.0])  # OpenSky with / without an account
def test_names_the_aircraft_that_flew_the_path(every_s):
    cfg = load_config().aircraft
    target = _flight((40, 200), 70, "3423c5", "IBE3456")
    # A decoy on a parallel route 3 km to one side, at the same height and speed.
    side = _flight((40, 200), 70, "3423c6", "VLG2002")
    east, north, _ = enu_basis(MADRID.latitude, MADRID.longitude)
    across = np.radians(70 + 90)
    side.xyz = side.xyz + 3000 * (np.sin(across) * east + np.cos(across) * north)
    log = _log_of([target, side, *_decoys(8, 1)], every_s)
    match = AircraftIdentifier(log, MADRID, cfg, "opensky").identify(
        _observe(target, clock_offset_s=1.5)
    )
    assert match is not None and match.callsign == "IBE3456" and match.icao24 == "3423c5"
    assert match.clock_offset_s == pytest.approx(1.5, abs=0.3)  # the camera clock error
    assert match.cross_track_deg < 0.1 and match.confidence >= 0.5
    assert match.altitude_m == pytest.approx(10_500, abs=100)
    assert match.speed_m_s == pytest.approx(230, abs=5)
    assert match.heading_deg == pytest.approx(70, abs=2)


def test_paths_no_aircraft_flew_are_not_named():
    """Satellite-like paths through a sky full of broadcasting aircraft stay unnamed."""
    cfg = load_config().aircraft
    ident = AircraftIdentifier(_log_of(_decoys(30, 2)), MADRID, cfg)
    rng = np.random.default_rng(3)
    for _ in range(90):
        start = altaz_to_vec(rng.uniform(10, 80), rng.uniform(0, 360))
        axis = np.cross(start, rng.normal(size=3))
        axis /= np.linalg.norm(axis)
        # deg/s: like nearby aircraft and satellites, and (one in three) distant aircraft
        speed = rng.uniform(0.2, 1.5) if rng.uniform() < 0.67 else rng.uniform(0.02, 0.2)
        s = np.arange(0, 20, 0.2)
        a = np.radians(speed * s)[:, None]
        v = start * np.cos(a) + np.cross(axis, start) * np.sin(a)
        alt, az = vec_to_altaz(v)
        t0 = EVENING + timedelta(seconds=float(rng.uniform(-60, 40)))
        times = [t0 + timedelta(seconds=float(x)) for x in s]
        path = SkyPath(times, np.zeros(len(s)), np.zeros(len(s)), alt, az)
        assert ident.identify(path) is None


def test_a_live_station_waits_for_broadcasts_after_the_pass():
    cfg = load_config().aircraft
    target = _flight((40, 200), 70, "3423c5", "IBE3456")
    path = _observe(target)  # ends 20 s after EVENING
    early = AircraftLog(
        target.broadcasts(EVENING - timedelta(seconds=60), 75, 10, 20, np.random.default_rng(0))
    )  # last broadcast at EVENING + 15 s
    assert not AircraftIdentifier(early, MADRID, cfg).covers(path)
    assert AircraftIdentifier(_log_of([target]), MADRID, cfg).covers(path)


def test_a_confident_match_makes_the_event_a_plane():
    event = SimpleNamespace(
        label=ObjectClass.SATELLITE,
        label_source="orbit",
        confidence=0.7,
        identification="STARLINK-1",
        identification_checked=True,
        aircraft=None,
        aircraft_checked=False,
    )
    match = SimpleNamespace(confidence=0.9)
    apply_match(event, match)  # type: ignore[arg-type]
    assert (event.label, event.label_source, event.confidence) == (ObjectClass.PLANE, "adsb", 0.9)
    assert event.identification is None and event.aircraft is match and event.aircraft_checked
    unsure = SimpleNamespace(**{**vars(event), "label": ObjectClass.SATELLITE})
    apply_match(unsure, SimpleNamespace(confidence=0.4))  # type: ignore[arg-type]
    assert unsure.label is ObjectClass.SATELLITE  # weak matches only annotate


# --------------------------------------------------------------------------- live feed
class _FakeSky:
    name = "opensky"

    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    def fetch(self):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return [_state("a", 1000.0 + self.calls)]


def test_feed_respects_the_daily_budget_and_rate_limits():
    cfg = load_config(aircraft={"max_requests_per_day": 3}).aircraft
    log, sky = AircraftLog(), _FakeSky()
    now = [EVENING.timestamp()]
    feed = AircraftFeed(sky, log, cfg, active=lambda: False, clock=lambda: now[0])
    assert not feed._wanted()  # nothing crossing the view: OpenSky isn't asked
    for _ in range(5):
        feed.poll()
    assert sky.calls == 3 and len(log) == 3 and "budget" in (feed.error or "")
    now[0] += 86_400  # a new day, a new budget
    feed.poll()
    assert sky.calls == 4

    limited = AircraftFeed(
        _FakeSky(fail=RateLimited(600)), AircraftLog(), cfg, clock=lambda: now[0]
    )
    limited.poll()
    limited.poll()  # still within the 600 s OpenSky asked us to wait
    assert limited.source.calls == 1 and "limit" in (limited.error or "")


# --------------------------------------------------------------------------- end to end
def _sky_video(path, catalog, seconds, fps, visible=None):
    """A simulated night over Madrid with one broadcasting aircraft crossing the view,
    plus the broadcasts of it and of 10 others (every second, like a local receiver)."""
    from skyseer.astro.calibrate import Calibration
    from skyseer.astro.simulate import camera_looking_at, render_aircraft_video

    cam = camera_looking_at(45, 200, 0, 960, 540, 55, k1=-0.03)
    start = EVENING - timedelta(seconds=seconds / 2)
    centre = remove_refraction(cam.pixel_to_vec(np.array([[480.0, 270.0]])))[0]
    target = flight_through(MADRID, centre, EVENING, 3500, 140, 95, "3423c5", "IBE3456")
    render_aircraft_video(
        path, cam, MADRID, start, seconds, fps, catalog, [target], visible=visible
    )
    rng = np.random.default_rng(5)
    states = []
    for f in [target, *_decoys(10, 7)]:
        states += f.broadcasts(start - timedelta(seconds=30), seconds + 60, 1.0, 10.0, rng)
    cal = Calibration(cam, start, MADRID, 100, 0.3)  # the true camera, as if star-calibrated
    return target, states, cal, start


def test_a_simulated_aircraft_is_named_in_a_recording(tmp_path, catalog):
    from skyseer.pipeline import run
    from skyseer.sources import FileSource

    path = tmp_path / "plane.mp4"
    target, states, cal, start = _sky_video(path, catalog, 20, 10)
    cfg = load_config()
    ident = AircraftIdentifier(AircraftLog(states), MADRID, cfg.aircraft)
    events = list(run(FileSource(path, start), cfg, calibration=cal, aircraft=ident))
    planes = [e for e in events if e.aircraft is not None]
    assert len(planes) == 1, [(e.label, e.label_source) for e in events]
    e = planes[0]
    assert (e.label, e.label_source, e.aircraft.callsign) == (ObjectClass.PLANE, "adsb", "IBE3456")
    # The strobe sits 3 px beside the steady light (a wingtip), which shifts the measured
    # path by ~0.1 deg: well inside the 0.5 deg acceptance limit.
    assert abs(e.aircraft.clock_offset_s) < 0.3 and e.aircraft.cross_track_deg < 0.2
    assert e.to_dict()["aircraft"]["name"] == "IBE3456"


class _ArrivingBroadcasts:
    """Broadcasts that reach SkySeer ``delay_s`` after they happen (replay time), like
    OpenSky's, which lag by several seconds."""

    name = "readsb"

    def __init__(self, states, start, delay_s):
        self.states, self.start, self.t0, self.delay_s = states, start.timestamp(), None, delay_s

    def fetch(self):
        import time

        if self.t0 is None:
            self.t0 = time.monotonic()
        now = self.start + (time.monotonic() - self.t0) - self.delay_s
        return [s for s in self.states if s.time <= now]


def test_a_live_station_names_the_aircraft_once_broadcasts_arrive(tmp_path, catalog):
    """The pass is shown at once, then identified when broadcasts after it have arrived
    (the station waits rather than guessing from too little)."""
    import json

    from skyseer.live import LiveStation
    from skyseer.sources import ReplaySource
    from skyseer.storage import Store

    path = tmp_path / "plane.mp4"
    target, states, cal, start = _sky_video(path, catalog, 20, 10, visible=[(1.0, 6.0)])
    cfg = load_config(
        aircraft={"check_delay_s": 1.0, "poll_s": 0.2, "max_clock_offset_s": 1.0},
        satellites={"enabled": False},
    )
    feed_log = AircraftLog()
    feed = AircraftFeed(_ArrivingBroadcasts(states, start, 6.0), feed_log, cfg.aircraft)
    ident = AircraftIdentifier(feed_log, MADRID, cfg.aircraft, "readsb")
    store = Store(tmp_path / "db.sqlite")

    class _Calibrated:  # stands in for the auto-calibrator: the camera is already solved
        calibration = cal

        def feed(self, frame):
            pass

        def camera_moved(self):
            pass

        def close(self):
            pass

    station = LiveStation(
        ReplaySource(path, start, speed=1.0),
        cfg,
        store,
        "replay",
        tmp_path / "night",
        MADRID,
        autocal=_Calibrated(),  # type: ignore[arg-type]
        clips=False,
        aircraft=ident,
        aircraft_feed=feed,
    )
    station.start()
    station.wait()
    lines = [json.loads(x) for x in (tmp_path / "night" / "events.jsonl").read_text().splitlines()]
    named = [x for x in lines if (x.get("aircraft") or {}).get("callsign") == "IBE3456"]
    assert named and named[0]["label"] == "Plane" and named[0]["label_source"] == "adsb"
    first = next(x for x in lines if x["id"] == named[0]["id"])
    assert "aircraft" not in first  # shown first, identified a little later
    stored = store.events(session_id=station.session_id, labels=["Plane"])
    assert stored and stored[0]["aircraft"]["callsign"] == "IBE3456"


def test_a_distant_slow_aircraft_is_named_without_a_clock_reading():
    """Far away near the horizon an aircraft crawls (~0.05 deg/s): its timing along the
    path can't be measured, so it is matched on position alone and the clock error is
    reported as unknown rather than as a meaningless number."""
    cfg = load_config().aircraft
    far = _flight((6, 200), 30, "4ca8af", "RYR2VR", altitude=11_500)  # ~120 km, approaching
    ident = AircraftIdentifier(_log_of([far, *_decoys(8, 4)]), MADRID, cfg)
    # ~1 px of centroid noise: a second of clock error moves it less than that.
    path = _observe(far, seconds=40, clock_offset_s=1.0, noise_deg=0.05)
    first, last = altaz_to_vec(path.alt[[0, -1]], path.az[[0, -1]])
    assert np.degrees(np.arccos(first @ last)) / 40 < 0.08  # deg/s
    match = ident.identify(path)
    assert match is not None and match.callsign == "RYR2VR"
    assert match.clock_offset_s is None and match.to_dict()["clock_offset_s"] is None
    # With a sharper measurement the same pass does time the clock (1.0 s off).
    sharp = ident.identify(_observe(far, seconds=40, clock_offset_s=1.0, noise_deg=0.005))
    assert sharp is not None and sharp.clock_offset_s == pytest.approx(1.0, abs=0.3)


def test_a_speck_next_to_a_distant_aircraft_is_not_named():
    """Too short a path (under min_arc_deg) can't show its direction: refuse to name it."""
    cfg = load_config().aircraft
    far = _flight((8, 200), 70, "4cabec", "EIN581", altitude=11_800)
    ident = AircraftIdentifier(_log_of([far]), MADRID, cfg)
    speck = _observe(far, seconds=2.0, noise_deg=0.02)  # 2 s of a ~0.15 deg/s aircraft
    assert ident.identify(speck) is None
    assert ident.identify(_observe(far, seconds=20.0, noise_deg=0.02)) is not None
