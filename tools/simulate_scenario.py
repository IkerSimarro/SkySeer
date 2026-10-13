"""Make a realistic test video for any place, time and camera direction, with known truth.

    # Real satellites crossing the view (uses the current orbit catalogue):
    python tools/simulate_scenario.py satellites --date 2026-10-14T20:00:00 \
        --lat 40.42 --lon -3.70 --alt 50 --az 250 --out data/sim/sats.mp4

    # Meteors from a shower plus sporadics (shower must be active on that date):
    python tools/simulate_scenario.py meteors --shower ORI --count 6 \
        --date 2026-10-21T23:30:00 --lat 40.42 --lon -3.70 --alt 50 --az 100 \
        --out data/sim/orionids.mp4

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
    ap.add_argument("kind", choices=["satellites", "meteors"])
    ap.add_argument("--date", required=True, help="UTC start, e.g. 2026-10-14T20:00:00")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--alt", type=float, default=50.0, help="camera altitude (deg)")
    ap.add_argument("--az", type=float, default=180.0, help="camera azimuth (deg, N=0, E=90)")
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

    start = datetime.fromisoformat(args.date)
    start = start if start.tzinfo else start.replace(tzinfo=UTC)
    obs = Observer(args.lat, args.lon, args.elevation)
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


if __name__ == "__main__":
    main()
