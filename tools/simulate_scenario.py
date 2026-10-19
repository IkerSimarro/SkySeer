"""Make a realistic test video for any place, time and camera direction, with known truth.

    # Real satellites crossing the view (uses the current orbit catalogue):
    python tools/simulate_scenario.py satellites --date 2026-10-14T20:00:00 \
        --lat 40.42 --lon -3.70 --alt 50 --az 250 --out data/sim/sats.mp4

    # Meteors from a shower plus sporadics (shower must be active on that date):
    python tools/simulate_scenario.py meteors --shower ORI --count 6 \
        --date 2026-10-21T23:30:00 --lat 40.42 --lon -3.70 --alt 50 --az 100 \
        --out data/sim/orionids.mp4

    # Real aircraft around you now: records ~1 minute of OpenSky broadcasts (about 10 of
    # the 400 free daily requests), points the camera where they fly and draws them there:
    python tools/simulate_scenario.py aircraft --lat 40.42 --lon -3.70 --seconds 40 \
        --out data/sim/aircraft.mp4

Writes the video and <name>.truth.json (what is really in it), then prints the
`skyseer analyze` command to run on it. Stars, satellites and meteors are placed with the
same astronomy SkySeer uses, so the result is only as independent as that maths; it tests
the whole pipeline end to end, not the astronomy formulas themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skyseer.astro.catalog import load_stars  # noqa: E402
from skyseer.astro.frames import Observer  # noqa: E402
from skyseer.astro.simulate import (  # noqa: E402
    camera_looking_at,
    find_passes,
    render_meteor_video,
    render_sky_video,
    shower_meteor,
    sporadic_meteor,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["satellites", "meteors", "aircraft"])
    ap.add_argument("--date", help="UTC start, e.g. 2026-10-14T20:00:00 (aircraft: now)")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--alt", type=float, default=None, help="camera altitude (deg)")
    ap.add_argument("--az", type=float, default=None, help="camera azimuth (deg, N=0, E=90)")
    ap.add_argument(
        "--log", type=Path, default=None, help="aircraft: reuse a recorded broadcast log"
    )
    ap.add_argument("--fov", type=float, default=55.0, help="horizontal field of view (deg)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--count", type=int, default=5, help="satellites or meteors to include")
    ap.add_argument("--shower", default=None, help="IAU code, e.g. PER, ORI, GEM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    obs = Observer(args.lat, args.lon, args.elevation)
    if args.kind == "aircraft":
        return aircraft_scenario(args, obs)
    if args.date is None:
        raise SystemExit("--date is needed for satellites and meteors")
    args.alt = 50.0 if args.alt is None else args.alt
    args.az = 180.0 if args.az is None else args.az

    start = datetime.fromisoformat(args.date)
    start = start if start.tzinfo else start.replace(tzinfo=UTC)
    cam = camera_looking_at(args.alt, args.az, 0, args.width, args.height, args.fov, k1=-0.05)
    stars = load_stars(7.0)
    rng = np.random.default_rng(args.seed)
    truth: dict = {
        "kind": args.kind,
        "start_utc": start.isoformat(),
        "observer": {
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "elevation_m": obs.elevation_m,
        },
        "camera": {"alt": args.alt, "az": args.az, "fov": args.fov},
        "objects": [],
    }

    if args.kind == "satellites":
        from skyseer.astro.orbits import load_catalog

        orbits = load_catalog(["active", "visual"], ROOT / "data" / "astro")
        passes = find_passes(orbits, obs, cam, start, args.seconds, min_seconds_in_view=6)
        if not passes:
            raise SystemExit(
                "No sunlit low-orbit satellites cross this view then; try "
                "1-2 hours after sunset or another direction."
            )
        chosen = [i for i, _ in passes[: args.count]]
        result = render_sky_video(
            args.out, cam, obs, start, args.seconds, args.fps, stars, orbits, chosen
        )
        truth["objects"] = [
            {
                "type": "satellite",
                "name": p.name.strip(),
                "norad_id": p.norad_id,
                "frames": len(p.points),
            }
            for p in result
        ]
    else:
        from skyseer.astro.showers import load_showers

        shower = None
        if args.shower:
            shower = next((s for s in load_showers() if s.code == args.shower.upper()), None)
            if shower is None:
                raise SystemExit(f"Unknown shower code {args.shower}")
        meteors = []
        for k in range(args.count):
            when = start + timedelta(seconds=3 + (args.seconds - 5) * k / max(args.count, 1))
            mag = float(rng.uniform(1.5, 2.5))
            m = None
            if shower is not None and k % 3 != 2:
                m = shower_meteor(
                    shower, when, obs, cam, rng, mag, radiant_spread_deg=0.7, speed_spread=0.1
                )
            meteors.append(m or sporadic_meteor(when, cam, rng, mag))
        render_meteor_video(args.out, cam, obs, start, args.seconds, args.fps, stars, meteors)
        truth["objects"] = [
            {
                "type": "meteor",
                "shower": m.shower or "sporadic",
                "start_utc": m.start_time.isoformat(),
                "peak_magnitude": round(m.peak_mag, 2),
                "frames": len(m.points),
            }
            for m in meteors
        ]

    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Video: {args.out}\nTruth: {truth_path}")
    for o in truth["objects"]:
        print("  ", {k: v for k, v in o.items() if k != "frames"})
    print(
        "\nAnalyse it with:\n  uv run skyseer analyze "
        f"{args.out.as_posix()} --start-time {start:%Y-%m-%dT%H:%M:%S} "
        f"--lat {args.lat} --lon {args.lon}"
    )


class LoggedFlight:
    """A real aircraft, flown along its recorded broadcasts (for the renderer)."""

    def __init__(self, icao24: str, states: list, callsign: str | None):
        self.icao24, self.states, self.callsign = icao24, states, callsign
        self.points: list[tuple[int, float, float]] = []

    def position(self, t: np.ndarray) -> np.ndarray:
        from skyseer.aircraft import trajectory

        xyz, ok = trajectory(self.states, np.asarray(t, dtype=float), 30.0, 15.0)
        xyz[~ok] = np.nan
        return xyz


def aircraft_scenario(args: argparse.Namespace, obs: Observer) -> None:
    """Record real broadcasts (or reuse a log), aim the camera where aircraft fly, render."""
    import time

    from skyseer.aircraft import (
        AircraftFeed,
        AircraftLog,
        OpenSkySource,
        enu_basis,
        geodetic_to_ecef,
    )
    from skyseer.astro.frames import vec_to_altaz
    from skyseer.astro.simulate import render_aircraft_video
    from skyseer.config import load_config

    cfg = load_config().aircraft
    log_path = args.out.with_suffix(".aircraft.jsonl")
    if args.log is not None:
        import shutil

        if args.log.resolve() != log_path.resolve():
            shutil.copyfile(args.log, log_path)
        broadcasts = AircraftLog.load(log_path)
    else:
        log_path.unlink(missing_ok=True)
        broadcasts = AircraftLog(path=log_path)
        feed = AircraftFeed(OpenSkySource(obs, cfg.radius_km), broadcasts, cfg)
        record_s = args.seconds + 40
        print(f"Recording OpenSky broadcasts for {record_s:.0f} s...")
        end = time.time() + record_s
        while time.time() < end:
            feed.poll()
            print(f"  {len(broadcasts):,} positions, {feed.requests_today} requests", flush=True)
            time.sleep(cfg.poll_s)
    span = broadcasts.span()
    if span is None:
        raise SystemExit("No aircraft broadcasts received.")
    # The last seconds of the recording: every aircraft's position is fresh there (OpenSky
    # also returns stale positions, minutes old, of aircraft it hasn't heard lately).
    t0 = span[1] - 5.0 - args.seconds
    start = datetime.fromtimestamp(t0, UTC)

    # Aim the camera at the aircraft that stays above 12 deg the longest (unless told).
    origin = geodetic_to_ecef(obs.latitude, obs.longitude, obs.elevation_m)
    to_enu = enu_basis(obs.latitude, obs.longitude)
    times = t0 + np.linspace(0, args.seconds, 21)
    flights, best = [], (0, None)
    for icao in broadcasts.ids_between(t0, t0 + args.seconds):
        states = broadcasts.states(icao)
        f = LoggedFlight(icao, states, next((s.callsign for s in states if s.callsign), None))
        enu = (f.position(times) - origin) @ to_enu.T
        enu /= np.linalg.norm(enu, axis=1)[:, None]
        alt, az = vec_to_altaz(enu)
        high = int(np.nansum(alt > 12))
        flights.append(f)
        if high > best[0]:
            best = (high, (float(alt[10]), float(az[10])))
    if args.alt is None or args.az is None:
        if best[1] is None:
            raise SystemExit("No aircraft above 12 degrees altitude; try later or elsewhere.")
        args.alt, args.az = best[1]
    cam = camera_looking_at(args.alt, args.az, 0, args.width, args.height, args.fov, k1=-0.05)
    stars = load_stars(7.0)
    render_aircraft_video(args.out, cam, obs, start, args.seconds, args.fps, stars, flights)
    seen = [f for f in flights if len(f.points) >= args.fps]  # at least a second in view
    truth = {
        "kind": "aircraft",
        "start_utc": start.isoformat(),
        "observer": {
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "elevation_m": obs.elevation_m,
        },
        "camera": {"alt": round(args.alt, 2), "az": round(args.az, 2), "fov": args.fov},
        "broadcast_log": log_path.as_posix(),
        "note": "Real OpenSky broadcasts drawn where they put each aircraft; the video is "
        "simulated, so a match shows the pipeline works, not that the broadcasts are right.",
        "objects": [
            {
                "type": "aircraft",
                "icao24": f.icao24,
                "callsign": f.callsign,
                "frames": len(f.points),
            }
            for f in seen
        ],
    }
    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Video: {args.out}\nTruth: {truth_path}\nBroadcasts: {log_path}")
    for o in truth["objects"]:
        print("  ", {k: v for k, v in o.items() if k != "frames"})
    print(
        "\nAnalyse it with:\n  uv run skyseer analyze "
        f"{args.out.as_posix()} --start-time {start:%Y-%m-%dT%H:%M:%S.%f} "
        f"--lat {args.lat} --lon {args.lon} --elevation {args.elevation} "
        f"--aircraft {log_path.as_posix()}"
    )


if __name__ == "__main__":
    main()
