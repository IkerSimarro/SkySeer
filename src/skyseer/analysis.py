"""Analysing a recorded video end to end: used by `skyseer analyze` and by dashboard
uploads, so both produce exactly the same results.

Steps: star calibration (optional), satellite catalogue, the pipeline with per-event
clips, then events.json/events.csv, calibration.json and run.json in the output folder,
which is registered in the station database.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
    from .aircraft import AircraftIdentifier
    from .astro.calibrate import Calibration
    from .astro.frames import Observer
    from .astro.satellites import SatelliteIdentifier

log = logging.getLogger(__name__)

Notify = Callable[[str], None]  # a human-readable status line ("Recognising stars...")
Progress = Callable[[int, int], None]  # (frames done, total frames)

HIDDEN = (ObjectClass.JUNK, ObjectClass.STAR)

# An analysis saves its progress as it goes, so a stopped one can carry on:
CHECKPOINT = "checkpoint.json"  # how far it got (removed when it finishes)
PARTIAL = "events.partial.jsonl"  # the events found so far, one per line
RESUME_WARMUP_S = 90.0  # video re-read before the resume point (background model, star
# filter context); nothing starting there is reported again


def can_resume(out_dir: Path) -> bool:
    """True if ``out_dir`` holds an analysis that was stopped before it finished."""
    return (out_dir / CHECKPOINT).is_file()


def _write_json(path: Path, data: Any) -> None:
    """Written whole or not at all (a stop mid-write leaves the previous version)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_partial(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / PARTIAL
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(json.JSONDecodeError):  # the last line, cut off by a stop
            events.append(json.loads(line))
    return events


def _forget_media(out_dir: Path, dropped: list[dict[str, Any]], next_id: int) -> None:
    """Delete the clips of events that will be found again, and any written after the
    last checkpoint (those numbers are given out again)."""
    for e in dropped:
        for rel in (e.get("media") or {}).values():
            if isinstance(rel, str):
                (out_dir / rel).unlink(missing_ok=True)
    for f in (out_dir / "clips").glob("*"):
        number = f.name[:4]
        if number.isdigit() and int(number) >= next_id:
            f.unlink(missing_ok=True)


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
    gray: bool = True,
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
        image, when = load_sky_image(video, start_time, 0.0, a.calibration_seconds, gray=gray)
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


def make_aircraft_identifier(
    cfg: Config,
    observer: Observer | None,
    log_path: Path | None,
    clock: float | None = None,
    notify: Notify = log.info,
) -> AircraftIdentifier | None:
    """Aircraft identifier from a recorded broadcast log (``aircraft.jsonl``, e.g. from
    ``skyseer aircraft record`` while filming), or None if there is none."""
    if log_path is None or not cfg.aircraft.enabled:
        return None
    if observer is None:
        notify("No location given: aircraft will not be named (add --lat/--lon).")
        return None
    from .aircraft import AircraftIdentifier, AircraftLog

    a = cfg.aircraft
    if clock is not None:
        a = a.model_copy(update={"max_clock_offset_s": clock})
    broadcasts = AircraftLog.load(log_path)
    span = broadcasts.span()
    if span is None:
        notify(f"No aircraft broadcasts in {log_path}; aircraft will not be named.")
        return None
    n_aircraft = len(broadcasts.ids_between(span[0], span[1]))
    notify(f"Loaded {len(broadcasts):,} broadcast positions of {n_aircraft} aircraft")
    return AircraftIdentifier(broadcasts, observer, a, "log")


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
            from .astro.moon import moon_over_night

            end = started + timedelta(seconds=seconds)
            info["moon"] = moon_over_night(started, end, obs, cal.camera)
    return info


def recording_time(
    video: Path, observer: Observer | None, notify: Notify = log.info
) -> datetime | None:
    """When a recording started, from the time its file says it was made, or None.

    That time is only trusted if a location is known and the Sun was down there for the
    whole recording: a downloaded or re-encoded video often carries the time it was
    processed instead (a YouTube video in testing said 12:50 UTC, the middle of the day).
    """
    import av

    from .astro.sun import sun_altitude

    try:
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            raw = c.metadata.get("creation_time") or stream.metadata.get("creation_time")
            seconds = float(c.duration / 1_000_000) if c.duration else 0.0
    except (av.error.FFmpegError, IndexError):
        return None
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return None
    when = when if when.tzinfo else when.replace(tzinfo=UTC)
    if when.year < 2000:  # a placeholder (1904 and 1970 are common)
        return None
    if observer is None:
        notify(
            f"The video file says it was made {when:%Y-%m-%d %H:%M} UTC, but without a "
            "location SkySeer can't check that it was night then; treating the time as "
            "unknown."
        )
        return None
    end = when + timedelta(seconds=seconds)
    if max(sun_altitude(when, observer), sun_altitude(end, observer)) > -6.0:
        notify(
            f"The video file says it was made {when:%Y-%m-%d %H:%M} UTC, but the Sun was up "
            "then at that place: probably when the file was processed, not recorded. "
            "Treating the time as unknown."
        )
        return None
    notify(f"Recording time from the video file: {when:%Y-%m-%d %H:%M:%S} UTC (night there)")
    return when


@dataclass
class Analysis:
    out_dir: Path
    events: list[Event]  # every track, including Junk and Star
    reported: list[Event]  # what was written to events.json
    calibration: Calibration | None
    identifier: SatelliteIdentifier | None
    stats: dict[str, Any] = field(default_factory=dict)
    session_id: int | None = None
    earlier: list[dict[str, Any]] = field(default_factory=list)  # found before a resume


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
    aircraft_log: Path | None = None,
    progress: Progress | None = None,
    notify: Notify = log.info,
    resume: bool = False,
    checkpoint_s: float = 300.0,
) -> Analysis:
    """Analyse ``video`` into ``out_dir``. Progress is saved every ``checkpoint_s`` of
    video; ``resume`` carries on a stopped analysis in ``out_dir`` from its last save,
    with the time, place and calibration it had."""
    state: dict[str, Any] | None = None
    if resume:
        if not can_resume(out_dir):
            raise ValueError(f"No stopped analysis to resume in {out_dir}")
        state = json.loads((out_dir / CHECKPOINT).read_text(encoding="utf-8"))
        start_time = datetime.fromisoformat(state["start_time"]) if state["start_time"] else None
        time_source = state["time_source"]
        place = state["observer"]
        if place is not None:
            from .astro.frames import Observer

            observer = Observer(**place)
        notify(f"Resuming from {timedelta(seconds=round(state['frame'] / state['fps']))}")
    else:
        time_source = "given"
        if start_time is None:
            start_time = recording_time(video, observer, notify)
            time_source = "video file" if start_time is not None else "unknown"
    if start_time is None:
        # Without the time, positions on the star background still work (they need only
        # time differences within the video), but nothing that needs the date or the
        # horizon does: never guess them from the file's date.
        notify(
            "The recording time is unknown: satellites, aircraft and meteor showers will "
            "not be identified. Give the start time to enable them."
        )
        observer = None
        off = {"enabled": False}
        cfg = cfg.model_copy(
            update={
                "satellites": cfg.satellites.model_copy(update=off),
                "showers": cfg.showers.model_copy(update=off),
                "aircraft": cfg.aircraft.model_copy(update=off),
            }
        )
    # Greyscale is all the analysis needs (and faster to decode), unless clips keep colour;
    # the calibration reads the video the same way.
    colour = clips and cfg.clips.enabled and cfg.clips.color
    cal = calibration
    if state is not None:
        from .astro.calibrate import Calibration

        saved = out_dir / "calibration.json"
        cal = Calibration.load(saved) if saved.exists() else None
    elif cal is None and sky:
        cal = calibrate_recording(video, cfg, observer, start_time, notify, gray=not colour)
    identifier = make_identifier(cfg, cal.observer, tle, clock, notify) if cal else None
    aircraft = (
        make_aircraft_identifier(cfg, cal.observer, aircraft_log, clock, notify) if cal else None
    )

    source = FileSource(video, start_time, gray=not colour)
    total = source.frame_count
    out_dir.mkdir(parents=True, exist_ok=True)
    if state is None:
        if cal is not None:
            cal.save(out_dir / "calibration.json")
        state = {
            "start_time": start_time.isoformat() if start_time else None,
            "time_source": time_source,
            "observer": None
            if observer is None
            else {
                "latitude": observer.latitude,
                "longitude": observer.longitude,
                "elevation_m": observer.elevation_m,
            },
            "fps": source.fps,
            "frame": 0,  # every event starting before this frame is in PARTIAL
            "next_event_id": 1,
            "disturbances": [],  # frame ranges, before ``frame``
            "discarded_tracks": 0,
        }
        _write_json(out_dir / CHECKPOINT, state)
        (out_dir / PARTIAL).unlink(missing_ok=True)
    keep_from = int(state["frame"])
    found = _read_partial(out_dir)
    earlier = [e for e in found if e["first_frame"] < keep_from]
    _forget_media(
        out_dir, [e for e in found if e["first_frame"] >= keep_from], state["next_event_id"]
    )
    source.from_s = max(0.0, keep_from / source.fps - RESUME_WARMUP_S) if keep_from else 0.0
    old_disturbances = [d for d in state["disturbances"] if d[0] < keep_from]
    old_discarded = int(state["discarded_tracks"])

    def save_progress(pipeline: Any, index: int) -> None:
        state.update(
            frame=pipeline.settled_frame(index),
            next_event_id=pipeline.next_event_id,
            disturbances=old_disturbances + [d for d in pipeline.disturbances if d[0] >= keep_from],
            # The re-read stretch before a resume point can count a few twice.
            discarded_tracks=old_discarded + pipeline.discarded_tracks,
        )
        _write_json(out_dir / CHECKPOINT, state)

    recorder = None
    if clips and cfg.clips.enabled:
        from .clips import DEFAULT_SKIP, ClipRecorder

        skip = frozenset() if include_junk else DEFAULT_SKIP
        recorder = ClipRecorder(out_dir, cfg.clips, source.fps, skip)
    stats: dict[str, Any] = {}
    notify("Analysing the video...")
    events: list[Event] = []
    tmp = out_dir / (PARTIAL + ".tmp")  # replaced whole: never left half-written
    tmp.write_text("".join(json.dumps(e) + "\n" for e in earlier), encoding="utf-8")
    tmp.replace(out_dir / PARTIAL)
    with open(out_dir / PARTIAL, "a", encoding="utf-8") as partial:
        for event in run(
            source,
            cfg,
            (lambda i: progress(i, total)) if progress else None,
            calibration=cal,
            identifier=identifier,
            stats=stats,
            clips=recorder,
            aircraft=aircraft,
            keep_from=keep_from,
            first_event_id=state["next_event_id"],
            checkpoint=save_progress,
            checkpoint_every=max(1, round(checkpoint_s * source.fps)),
        ):
            events.append(event)
            if include_junk or event.label not in HIDDEN:
                partial.write(json.dumps(event.to_dict()) + "\n")
                partial.flush()
    if progress:
        progress(total, total)
    if keep_from:
        fps = source.fps
        stats["disturbances"] = [(s / fps, e / fps) for s, e in old_disturbances] + [
            d for d in stats["disturbances"] if d[0] * fps >= keep_from
        ]
        stats["discarded_tracks"] += old_discarded

    reported = events if include_junk else [e for e in events if e.label not in HIDDEN]
    write_reports(earlier + [e.to_dict() for e in reported], out_dir)
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
    info["time_source"] = time_source  # "given", "video file" or "unknown"
    (out_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    session_id = store.import_run(out_dir) if store is not None else None
    (out_dir / CHECKPOINT).unlink(missing_ok=True)  # finished: nothing left to resume
    (out_dir / PARTIAL).unlink(missing_ok=True)
    return Analysis(out_dir, events, reported, cal, identifier, stats, session_id, earlier)
