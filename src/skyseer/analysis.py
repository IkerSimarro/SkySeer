"""Analysing a recorded video end to end: used by `skyseer analyze` and by dashboard
uploads, so both produce exactly the same results.

Steps: star calibration (optional), satellite catalogue, the pipeline with per-event
clips, then events.json/events.csv, calibration.json and run.json in the output folder,
which is registered in the station database.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .config import Config
from .models import Event, ObjectClass
from .pipeline import run
from .reports import write_reports
from .sources import FileSource
from .storage import Store

if TYPE_CHECKING:
    from .astro.calibrate import Calibration
    from .astro.frames import Observer
    from .astro.satellites import SatelliteIdentifier

log = logging.getLogger(__name__)

Notify = Callable[[str], None]  # a human-readable status line ("Recognising stars...")
Progress = Callable[[int, int], None]  # (frames done, total frames)

HIDDEN = (ObjectClass.JUNK, ObjectClass.STAR)


def is_simulation(source: str) -> bool:
    """True for videos made by SkySeer's simulators, which write their truth alongside."""
    video = Path(source)
    return video.with_suffix(".truth.json").exists() or (
        video.with_suffix(".json").exists() and video.suffix == ".mp4"
    )


def calibrate_recording(
    video: Path,
    cfg: Config,
    observer: Observer | None,
    start_time: datetime | None,
    notify: Notify = log.info,
) -> Calibration | None:
    """Calibrate from the first seconds of a recording; None (with a message) on failure."""
    from .astro.calibrate import CalibrationError, calibrate, load_sky_image
    from .astro.catalog import build_database

    a = cfg.astro
    try:
        notify("Preparing the star database (the first run downloads and builds it)...")
        db = build_database(
            a.database_min_fov, a.database_max_fov, a.database_max_mag, Path(a.data_dir)
        )
        notify("Recognising stars to calibrate the camera...")
        image, when = load_sky_image(video, start_time, 0.0, a.calibration_seconds)
        cal = calibrate(
            image,
            when,
            observer,
            fov_estimate=a.fov_estimate,
            fov_max_error=a.fov_tolerance if a.fov_estimate else None,
            database=db,
        )
    except CalibrationError as e:
        notify(f"Camera calibration failed ({e}); continuing without sky coordinates.")
        return None
    s = cal.summary()
    where = (
        f"az {s['pointing_az']:.1f}°, alt {s['pointing_alt']:.1f}°"
        if observer is not None
        else f"RA {s['center_ra']:.1f}°, Dec {s['center_dec']:+.1f}°"
    )
    notify(
        f"Camera calibrated from {cal.n_stars} stars: pointing {where}, "
        f"field {s['fov_h']:.1f}° x {s['fov_v']:.1f}°"
    )
    return cal


def make_identifier(
    cfg: Config,
    observer: Observer | None,
    tle: list[Path] | None = None,
    clock: float | None = None,
    notify: Notify = log.info,
) -> SatelliteIdentifier | None:
    """Satellite identifier from the orbit catalogue, or None if it can't be used."""
    if observer is None or not cfg.satellites.enabled:
        if cfg.satellites.enabled:
            notify("No location given: satellites will not be named (add --lat/--lon).")
        return None
    from .astro.orbits import load_catalog
    from .astro.satellites import SatelliteIdentifier

    s = cfg.satellites
    if clock is not None:
        s = s.model_copy(update={"max_clock_offset_s": clock})
    extra = [Path(p) for p in s.extra_tle_files] + list(tle or [])
    try:
        notify("Loading satellite orbits (downloads from CelesTrak if stale)...")
        orbits = load_catalog(s.groups, Path(cfg.astro.data_dir), extra, s.max_age_hours)
    except RuntimeError as e:
        notify(f"Satellite orbits unavailable ({e}); satellites will not be named.")
        return None
    notify(f"Loaded orbits of {len(orbits):,} satellites")
    return SatelliteIdentifier(orbits, observer, s)


def run_info(
    kind: str,
    source: str,
    started: datetime,
    seconds: float,
    fps: float,
    frame_size: tuple[int, int],
    cal: Calibration | None,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """What run.json records about a session (read back by storage.Store.import_run)."""
    info: dict[str, Any] = {
        "kind": kind,
        "source": source,
        "started": started.isoformat(),
        "ended": (started + timedelta(seconds=seconds)).isoformat(),
        "fps": fps,
        "frame_size": list(frame_size),
        "skyseer_version": __version__,
        "simulated": is_simulation(source),
        "disturbances": [[round(a, 2), round(b, 2)] for a, b in stats.get("disturbances", [])],
        "discarded_tracks": stats.get("discarded_tracks", 0),
    }
    if cal is not None:
        info["calibration"] = cal.summary()
        if cal.observer is not None:
            obs = cal.observer
            info["observer"] = {
                "lat": obs.latitude,
                "lon": obs.longitude,
                "elevation_m": obs.elevation_m,
            }
            info["pointing"] = cal.camera.pointing_altaz()
    return info


@dataclass
class Analysis:
    out_dir: Path
    events: list[Event]  # every track, including Junk and Star
    reported: list[Event]  # what was written to events.json
    calibration: Calibration | None
    identifier: SatelliteIdentifier | None
    stats: dict[str, Any] = field(default_factory=dict)
    session_id: int | None = None


def analyze_video(
    video: Path,
    cfg: Config,
    out_dir: Path,
    start_time: datetime | None = None,
    observer: Observer | None = None,
    calibration: Calibration | None = None,
    sky: bool = True,
    tle: list[Path] | None = None,
    clock: float | None = None,
    clips: bool = True,
    include_junk: bool = False,
    store: Store | None = None,
    progress: Progress | None = None,
    notify: Notify = log.info,
) -> Analysis:
    cal = calibration
    if cal is None and sky:
        cal = calibrate_recording(video, cfg, observer, start_time, notify)
    identifier = make_identifier(cfg, cal.observer, tle, clock, notify) if cal else None

    source = FileSource(video, start_time)
    total = source.frame_count
    recorder = None
    if clips and cfg.clips.enabled:
        from .clips import DEFAULT_SKIP, ClipRecorder

        skip = frozenset() if include_junk else DEFAULT_SKIP
        recorder = ClipRecorder(out_dir, cfg.clips, source.fps, skip)
    stats: dict[str, Any] = {}
    notify("Analysing the video...")
    events = list(
        run(
            source,
            cfg,
            (lambda i: progress(i, total)) if progress else None,
            calibration=cal,
            identifier=identifier,
            stats=stats,
            clips=recorder,
        )
    )
    if progress:
        progress(total, total)

    reported = events if include_junk else [e for e in events if e.label not in HIDDEN]
    write_reports([e.to_dict() for e in reported], out_dir)
    if cal is not None:
        cal.save(out_dir / "calibration.json")
    info = run_info(
        "file",
        str(video),
        source.start_time,
        total / source.fps,
        source.fps,
        (source.width, source.height),
        cal,
        stats,
    )
    (out_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    session_id = store.import_run(out_dir) if store is not None else None
    return Analysis(out_dir, events, reported, cal, identifier, stats, session_id)
