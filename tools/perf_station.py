"""Throughput at the Raspberry Pi station's camera settings.

    python tools/perf_station.py [--seconds 30] [--width 2028 --height 1520 --fps 10]

Renders a synthetic night at the station's resolution and frame rate (cached in
data/perf/), then measures frames per second for the bare pipeline and for the whole
live station (clips, live preview, database), at full and half processing resolution.
A station keeps up when it processes at least as many frames per second as the camera
delivers.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from skyseer.config import load_config
from skyseer.live import LiveStation
from skyseer.pipeline import run
from skyseer.sources import FileSource, ReplaySource
from skyseer.storage import Store
from skyseer.synth import SceneSpec, generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--width", type=int, default=2028)
    ap.add_argument("--height", type=int, default=1520)
    ap.add_argument("--fps", type=float, default=10)
    args = ap.parse_args()
    out = Path("data/perf")
    video = out / f"night_{args.width}x{args.height}_{args.fps:g}fps_{args.seconds:g}s.mp4"
    if not video.exists():
        print(f"Rendering {video} ...", flush=True)
        area = args.width * args.height / (960 * 540)
        generate(
            SceneSpec(
                width=args.width,
                height=args.height,
                fps=args.fps,
                seconds=args.seconds,
                seed=7,
                n_stars=int(250 * area),
                n_satellites=3,
                n_meteors=2,
                n_planes=1,
            ),
            video,
        )
    frames = int(args.seconds * args.fps)
    print(f"CPU: {platform.processor() or platform.machine()}; camera rate {args.fps:g} fps")
    for scale in (1.0, 0.5):
        cfg = load_config(detector={"process_scale": scale})
        t0 = time.perf_counter()
        events = list(run(FileSource(video), cfg))
        fps = frames / (time.perf_counter() - t0)
        real = sum(e.label not in ("Junk", "Star") for e in events)
        print(f"pipeline       process_scale {scale}: {fps:6.1f} fps ({real} objects)", flush=True)
        store = Store(out / "perf.db")
        night = LiveStation(ReplaySource(video, speed=0), cfg, store, "perf", out / f"live_{scale}")
        t0 = time.perf_counter()
        night.start()
        night.wait()
        fps = night.frames / (time.perf_counter() - t0)
        print(
            f"live station   process_scale {scale}: {fps:6.1f} fps (clips + preview + database)",
            flush=True,
        )


if __name__ == "__main__":
    main()
