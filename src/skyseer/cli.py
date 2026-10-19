"""Command-line interface: ``skyseer analyze | calibrate | live | synth | bench``."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .models import Event
from .render import render_annotated_video
from .sources import CameraSource

if TYPE_CHECKING:  # the astronomy stack is imported lazily to keep the CLI quick to start
    from .astro.frames import Observer
    from .astro.satellites import SatelliteIdentifier
    from .live import LiveStation
    from .storage import Store

app = typer.Typer(
    help="SkySeer: satellite and meteor detection for night-sky video.", no_args_is_help=True
)


def _utf8_output() -> None:
    """Write UTF-8 even when output goes to a file, pipe or service log: on Windows those
    default to an old code page that can't encode the progress spinner (it crashed)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_utf8_output()
console = Console()

# Accepted time formats: seconds may have decimals; an explicit offset (+02:00) overrides UTC.
TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
]
ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _config(path: Path | None, sensitivity: int | None, frame_skip: int | None) -> Config:
    overrides: dict = {}
    if sensitivity is not None:
        overrides["detector"] = {"sensitivity": sensitivity}
    if frame_skip is not None:
        overrides["pipeline"] = {"frame_skip": frame_skip}
    return load_config(path, **overrides)


def _summary(events: list[Event]) -> Table:
    table = Table(title=f"{len(events)} events", header_style="bold cyan")
    with_sky = any(e.sky is not None for e in events)
    with_mag = any(e.photometry is not None for e in events)
    with_id = any(
        e.identification_checked or e.shower_checked or e.aircraft_checked for e in events
    )
    cols = ["#", "Class", "Conf.", "Start (UTC)", "Duration"]
    cols += ["Speed °/s", "Arc °"] if with_sky else ["Speed px/f", "Linearity"]
    if with_mag:
        cols.append("Mag")
    if with_id:
        cols.append("Identified as")
    for col in cols:
        table.add_column(col, justify="left" if col in ("Class", "Identified as") else "right")
    for e in events:
        row = [
            str(e.id),
            str(e.label),
            f"{e.confidence:.2f}",
            e.track.start_time.strftime("%H:%M:%S"),
            f"{e.track.duration:.1f}s",
        ]
        if with_sky:
            row += (
                [f"{e.sky.angular_speed:.2f}", f"{e.sky.arc_deg:.1f}"]
                if e.sky is not None
                else ["-", "-"]
            )
        else:
            row += [f"{e.features['avg_speed']:.2f}", f"{e.features['linearity']:.2f}"]
        if with_mag:
            row.append(e.photometry.peak_label if e.photometry is not None else "-")
        if with_id:
            row.append(_identity(e))
        table.add_row(*row)
    return table


def _identity(e: Event) -> str:
    """What the event is beyond its class: satellite name, meteor shower, fireball."""
    parts = []
    if e.aircraft is not None:
        a = e.aircraft
        parts.append(f"{a.short_label()}, {a.altitude_m:,.0f} m")
    elif e.aircraft_checked and e.label == "Plane":
        parts.append("[dim]no broadcast match[/dim]")
    if e.identification is not None:
        parts.append(e.identification.short_label())
    elif e.identification_checked and e.label == "Satellite":
        parts.append("[dim]no catalogued match[/dim]")
    if e.shower_checked:
        parts.append(e.shower.short_label() if e.shower is not None else "sporadic")
    if e.periodicity is not None:
        parts.append(f"blinks every {e.periodicity.period_s:.1f}s")
    if e.fireball:
        parts.append("[bold red]FIREBALL[/bold red]")
    return ", ".join(parts)


LatOpt = Annotated[float | None, typer.Option("--lat", help="Observer latitude, degrees N")]
LonOpt = Annotated[float | None, typer.Option("--lon", help="Observer longitude, degrees E")]
ElevOpt = Annotated[float | None, typer.Option("--elevation", help="Observer elevation, m")]
ReplayOpt = Annotated[
    Path | None,
    typer.Option(exists=True, dir_okay=False, help="Play this video as if it were the live camera"),
]
SpeedOpt = Annotated[
    float, typer.Option(help="Replay speed (1 = real time, 0 = as fast as possible)")
]
RecordedOpt = Annotated[
    datetime | None,
    typer.Option(
        "--start-time", help="Replay only: when the video was recorded (UTC)", formats=TIME_FORMATS
    ),
]


def _observer(
    cfg: Config, lat: float | None, lon: float | None, elevation: float | None
) -> Observer | None:
    """Observer from CLI options, falling back to the config file; None if unknown."""
    from .astro.frames import Observer

    o = cfg.observer
    lat = o.latitude if lat is None else lat
    lon = o.longitude if lon is None else lon
    if lat is None or lon is None:
        return None
    return Observer(lat, lon, o.elevation_m if elevation is None else elevation)


def _warn_if_no_dashboard() -> None:
    from .api.app import STATIC_DIR

    if not (STATIC_DIR / "index.html").exists():
        console.print(
            "[yellow]The dashboard page isn't built yet (the API works): "
            "cd web; npm ci; npm run build[/yellow]"
        )


def _say(message: str) -> None:
    """Status lines from the shared analysis code; problems in yellow."""
    bad = ("failed", "unavailable", "No location", "will not")
    console.print(f"[yellow]{message}[/yellow]" if any(b in message for b in bad) else message)


TleOpt = Annotated[
    list[Path] | None,
    typer.Option("--tle", help="Extra TLE file(s), e.g. historical orbits for an old video"),
]
ClockOpt = Annotated[
    float | None,
    typer.Option("--clock-tolerance", help="How many seconds the camera clock may be off"),
]
AircraftOpt = Annotated[
    str | None,
    typer.Option(
        "--aircraft",
        help="Name aircraft from live broadcasts: opensky (free online), readsb (a local "
        "receiver) or none (default: config)",
    ),
]
AircraftLogOpt = Annotated[
    Path | None,
    typer.Option(
        "--aircraft-log",
        exists=True,
        dir_okay=False,
        help="Replay only: aircraft broadcasts recorded with the video",
    ),
]


def _identifier(
    cfg: Config, observer: Observer | None, tle: list[Path] | None, clock: float | None
) -> SatelliteIdentifier | None:
    from .analysis import make_identifier

    return make_identifier(cfg, observer, tle, clock, _say)


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory")] = None,
    config: ConfigOpt = None,
    start_time: Annotated[
        datetime | None,
        typer.Option(
            help="Recording start, UTC unless a zone is given (2026-09-30T21:15:00.5)",
            formats=TIME_FORMATS,
        ),
    ] = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    calibration: Annotated[
        Path | None, typer.Option(help="Use a saved calibration.json instead of solving")
    ] = None,
    sky: Annotated[
        bool, typer.Option("--sky/--no-sky", help="Calibrate from the stars and add sky positions")
    ] = True,
    tle: TleOpt = None,
    clock_tolerance: ClockOpt = None,
    aircraft: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Aircraft broadcasts recorded while filming (skyseer aircraft record)",
        ),
    ] = None,
    sensitivity: Annotated[int | None, typer.Option(min=1, max=10)] = None,
    frame_skip: Annotated[int | None, typer.Option(min=1)] = None,
    video_out: Annotated[
        bool, typer.Option("--video/--no-video", help="Render annotated video")
    ] = True,
    clips: Annotated[
        bool, typer.Option("--clips/--no-clips", help="Write a clip and images per event")
    ] = True,
    include_junk: Annotated[bool, typer.Option(help="Also report Junk and Star events")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Analyse a recorded video: detect, calibrate from the stars, name satellites.

    Writes events.json, events.csv, calibration.json, an annotated video and, in clips/,
    a short clip, trail image and context image for each event.
    """
    _setup_logging(verbose)
    cfg = _config(config, sensitivity, frame_skip)
    out_dir = out or Path("data/runs") / f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    when = (
        None
        if start_time is None
        else (start_time if start_time.tzinfo else start_time.replace(tzinfo=UTC))
    )
    observer = _observer(cfg, lat, lon, elevation)

    from .analysis import analyze_video

    cal = None
    if calibration is not None:
        from .astro.calibrate import Calibration

        cal = Calibration.load(calibration)
    with Progress(
        TextColumn("[cyan]Analysing"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=None)
        result = analyze_video(
            video,
            cfg,
            out_dir,
            when,
            observer,
            cal,
            sky,
            tle,
            clock_tolerance,
            clips,
            include_junk,
            store=_store(cfg),
            aircraft_log=aircraft,
            progress=lambda i, n: progress.update(task, completed=i, total=n or None),
            notify=_say,
        )
    events, reported, stats = result.events, result.reported, result.stats
    cal, identifier = result.calibration, result.identifier
    if video_out and reported:
        predictions = {}
        if identifier is not None and cal is not None:
            for e in reported:
                if e.identification is not None:
                    predictions[e.id] = identifier.predicted_pixels(
                        e.identification, cal.camera, e.track.start_time, e.track.end_time
                    )
        with console.status("Rendering annotated video..."):
            render_annotated_video(
                video, reported, out_dir / "annotated.mp4", predictions=predictions
            )

    counts = Counter(str(e.label) for e in events)
    console.print(_summary(reported))
    console.print("All tracks by class: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if stats.get("disturbances"):
        spans = ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in stats["disturbances"][:8])
        more = " ..." if len(stats["disturbances"]) > 8 else ""
        console.print(
            f"Ignored {len(stats['disturbances'])} camera movement(s) / scene "
            f"change(s) at {spans}{more} ({stats['discarded_tracks']} tracks discarded)"
        )
    console.print(f"Results written to [bold]{out_dir}[/bold]")


def _store(cfg: Config) -> Store:
    """The station database (the SKYSEER_DATABASE environment variable overrides the config)."""
    import os

    from .storage import Store

    return Store(os.environ.get("SKYSEER_DATABASE") or cfg.storage.database)


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


@app.command()
def live(
    camera: Annotated[str, typer.Option(help="Camera index or stream URL")] = "0",
    replay: ReplayOpt = None,
    start_time: RecordedOpt = None,
    speed: SpeedOpt = 1.0,
    config: ConfigOpt = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    sky: Annotated[
        bool, typer.Option("--sky/--no-sky", help="Keep the camera calibrated from the stars")
    ] = True,
    tle: TleOpt = None,
    clock_tolerance: ClockOpt = None,
    aircraft: AircraftOpt = None,
    aircraft_log: AircraftLogOpt = None,
    clips: Annotated[
        bool, typer.Option("--clips/--no-clips", help="Write a clip and images per event")
    ] = True,
    verbose: VerboseOpt = False,
) -> None:
    """Run the station on a live camera and print events as they happen (Ctrl+C to stop).

    Everything is stored as with `skyseer serve --camera` (the database, plus
    data/live/<night>/ with clips, events.jsonl and events.json), so the dashboard
    shows it afterwards.
    """
    _setup_logging(verbose)
    cfg = load_config(config)

    def show(e: Event) -> None:
        where = f", {e.sky.angular_speed:.2f}°/s" if e.sky is not None else ""
        name = f" = {e.identification.short_label()}" if e.identification else ""
        if e.aircraft is not None:
            name = f" = {e.aircraft.short_label()}"
        console.print(
            f"[bold]{e.track.start_time:%H:%M:%S}[/bold] #{e.id} {e.label}{name} "
            f"({e.confidence:.2f}, {e.track.duration:.1f}s{where})"
        )

    station = _live_station(
        cfg,
        camera,
        replay,
        start_time,
        speed,
        _observer(cfg, lat, lon, elevation),
        sky,
        tle,
        clock_tolerance,
        clips,
        show,
        aircraft,
        aircraft_log,
    )
    station.start()
    console.print(
        f"Watching {station.camera} at {station.source.width}x{station.source.height}, "
        f"{station.source.fps:.0f} fps (Ctrl+C to stop)"
    )
    try:
        station.wait()
    except KeyboardInterrupt:
        console.print("Stopping...")
    finally:
        station.stop()
    console.print(f"Session saved in [bold]{station.folder}[/bold]")


def _live_station(
    cfg: Config,
    camera: str | None,
    replay: Path | None,
    start_time: datetime | None,
    speed: float,
    observer: Observer | None,
    sky: bool,
    tle: list[Path] | None,
    clock_tolerance: float | None,
    clips: bool,
    on_event: Any = None,
    aircraft: str | None = None,
    aircraft_log: Path | None = None,
) -> LiveStation:
    """A live station on a camera, or on a replayed video (for demos and testing)."""
    from .analysis import is_simulation
    from .live import LiveStation
    from .sources import ReplaySource

    source: Any
    if replay is not None:
        when = start_time
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        source = ReplaySource(replay, when, speed)
        label, kind = f"replay of {replay.name}", "replay"
    elif camera and camera.startswith("picamera"):  # picamera or picamera:1
        from .pi_camera import PiCameraSettings, PiCameraSource

        c = cfg.camera
        number = int(camera.split(":")[1]) if ":" in camera else 0
        settings = PiCameraSettings(
            width=c.width or 2028,
            height=c.height or 1520,
            fps=c.fps,
            exposure_us=round(c.exposure_ms * 1000) if c.exposure_ms else None,
            gain=c.gain,
            colour_gains=c.colour_gains,
            camera_num=number,
        )
        source = PiCameraSource(settings)
        label, kind = f"Pi camera {number}", "live"
    else:
        device: int | str = int(camera) if camera and camera.isdigit() else (camera or 0)
        source = CameraSource(device, cfg.camera.width, cfg.camera.height)
        label, kind = f"camera {camera}", "live"
    autocal = identifier = None
    if sky:
        from .astro.autocal import AutoCalibrator

        # A fixed camera keeps its calibration between nights; a replay must not reuse it.
        cache = None if replay else Path(cfg.astro.data_dir) / "live_calibration.json"
        autocal = AutoCalibrator(cfg.astro, observer, source.fps, cache_path=cache)
        identifier = _identifier(cfg, observer, tle, clock_tolerance)
    folder = Path(cfg.storage.live_dir) / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}"
    planes, feed = _aircraft_for(cfg, observer if sky else None, replay, aircraft, aircraft_log)
    if feed is not None:
        feed.log.path = folder / "aircraft.jsonl"  # the night's broadcasts are kept with it
    return LiveStation(
        source,
        cfg,
        _store(cfg),
        label,
        folder,
        observer,
        autocal,
        identifier,
        clips,
        on_event,
        simulated=replay is not None and is_simulation(str(replay)),
        aircraft=planes,
        aircraft_feed=feed,
    )


def _aircraft_for(
    cfg: Config,
    observer: Observer | None,
    replay: Path | None,
    source: str | None,
    log_path: Path | None,
) -> tuple[Any, Any]:
    """(AircraftIdentifier, AircraftFeed) for a live station, or Nones."""
    from .aircraft import AircraftFeed, AircraftIdentifier, AircraftLog, make_source

    a = cfg.aircraft
    kind = source or a.source
    if not a.enabled or observer is None or (kind == "none" and log_path is None):
        if kind != "none" and a.enabled and observer is None:
            _say("No location given: aircraft will not be named (add --lat/--lon).")
        return None, None
    if log_path is not None:  # a replay with the broadcasts recorded at the time
        return AircraftIdentifier(AircraftLog.load(log_path), observer, a, "log"), None
    if replay is not None:
        _say("A replay is in the past, so live broadcasts will not match (use --aircraft-log).")
        return None, None
    feed = AircraftFeed(make_source(a, observer, kind), AircraftLog(keep_s=2 * 3600), a)
    _say(f"Naming aircraft from {kind} broadcasts")
    return AircraftIdentifier(feed.log, observer, a, kind), feed


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option(help="127.0.0.1 = this computer only; 0.0.0.0 = your whole network"),
    ] = "127.0.0.1",
    port: int = 8000,
    camera: Annotated[
        str | None, typer.Option(help="Also run the live station on this camera (index or URL)")
    ] = None,
    replay: ReplayOpt = None,
    start_time: RecordedOpt = None,
    speed: SpeedOpt = 1.0,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    sky: Annotated[
        bool, typer.Option("--sky/--no-sky", help="Keep the camera calibrated from the stars")
    ] = True,
    tle: TleOpt = None,
    clock_tolerance: ClockOpt = None,
    aircraft: AircraftOpt = None,
    aircraft_log: AircraftLogOpt = None,
    config: ConfigOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Start the dashboard and its API (then open http://localhost:8000).

    With --camera (or --replay VIDEO, to demonstrate without a camera) the live station
    runs too, and the dashboard's Live page shows it as it happens.
    """
    import uvicorn

    from .api import create_app

    _setup_logging(verbose)
    cfg = load_config(config)
    station = None
    if camera is not None or replay is not None:
        station = _live_station(
            cfg,
            camera,
            replay,
            start_time,
            speed,
            _observer(cfg, lat, lon, elevation),
            sky,
            tle,
            clock_tolerance,
            clips=True,
            aircraft=aircraft,
            aircraft_log=aircraft_log,
        )
    store = _store(cfg)
    added = store.import_runs(cfg.storage.runs_dir, cfg.storage.live_dir)
    if added:
        console.print(f"Imported {added} earlier run(s) from {cfg.storage.runs_dir}")
    if host not in ("127.0.0.1", "localhost"):
        console.print(
            "[yellow]Listening on the network: anyone on it can open the dashboard "
            "(there is no login).[/yellow]"
        )
    console.print(f"Dashboard: [bold]http://localhost:{port}[/bold]  (API docs at /api/docs)")
    _warn_if_no_dashboard()
    if station is not None:
        station.start()
        console.print(f"Live station running on {station.camera}")
    try:
        uvicorn.run(create_app(cfg, store, station), host=host, port=port, log_level="warning")
    finally:
        if station is not None:
            station.stop()


@app.command()
def station(
    host: Annotated[
        str,
        typer.Option(help="127.0.0.1 = this computer only; 0.0.0.0 = your whole network"),
    ] = "127.0.0.1",
    port: int = 8000,
    camera: Annotated[
        str | None, typer.Option(help="Camera index, stream URL or picamera (default: config)")
    ] = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    sky: Annotated[
        bool, typer.Option("--sky/--no-sky", help="Keep the camera calibrated from the stars")
    ] = True,
    tle: TleOpt = None,
    clock_tolerance: ClockOpt = None,
    gps: Annotated[
        str | None,
        typer.Option(help="GPS: gpsd, a device such as /dev/ttyACM0, or none (default: config)"),
    ] = None,
    aircraft: AircraftOpt = None,
    config: ConfigOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run the unattended station: observe every night from dusk to dawn, with the
    dashboard always available (this is what the Raspberry Pi service runs)."""
    import uvicorn

    from .api import create_app
    from .gps import GpsMonitor
    from .station import Station

    _setup_logging(verbose)
    cfg = load_config(config)
    cam = camera or cfg.station.camera
    observer = _observer(cfg, lat, lon, elevation)
    source = gps or cfg.gps.source
    monitor = None
    if source != "none":
        g = cfg.gps
        monitor = GpsMonitor(source, g.cache_file, g.gpsd_host, g.gpsd_port, g.min_satellites)
    if observer is None and monitor is None and not cfg.station.always_on:
        console.print(
            "[yellow]No location given: SkySeer can't tell when it's dark, so it observes "
            "around the clock. Add --lat/--lon, a GPS (--gps gpsd) or observer: in the "
            "config.[/yellow]"
        )
    store = _store(cfg)
    store.import_runs(cfg.storage.runs_dir, cfg.storage.live_dir)
    st: Station

    def make_night() -> LiveStation:  # each night uses the station's latest location
        return _live_station(
            cfg,
            cam,
            None,
            None,
            1.0,
            st.observer,
            sky,
            tle,
            clock_tolerance,
            True,
            aircraft=aircraft,
        )

    st = Station(cfg, store, make_night, observer, gps=monitor)
    st.start()
    console.print(
        f"Station running on camera {cam}; observing while the Sun is below "
        f"{cfg.station.dark_sun_altitude:.0f}°. Dashboard: [bold]http://localhost:{port}[/bold]"
    )
    _warn_if_no_dashboard()
    try:
        uvicorn.run(create_app(cfg, store, st), host=host, port=port, log_level="warning")
    finally:
        st.stop()


@app.command()
def synth(
    out: Annotated[Path, typer.Argument(help="Output .mp4 path (ground truth saved alongside)")],
    seconds: float = 20.0,
    width: int = 960,
    height: int = 540,
    fps: float = 30.0,
    seed: int = 0,
) -> None:
    """Generate a synthetic night-sky video with known satellites, meteors and planes."""
    from .synth import SceneSpec, generate

    truth = generate(
        SceneSpec(width=width, height=height, fps=fps, seconds=seconds, seed=seed), out
    )
    console.print(
        f"Wrote {out} with {len(truth.objects)} objects: "
        + ", ".join(f"{o.label}" for o in truth.objects)
    )


@app.command()
def bench(
    scenes: Annotated[int, typer.Option(help="Number of synthetic scenes")] = 10,
    seconds: float = 20.0,
    config: ConfigOpt = None,
    frame_skip: Annotated[int | None, typer.Option(min=1)] = None,
    cache: Annotated[Path, typer.Option(help="Where scenes are stored")] = Path("data/bench"),
) -> None:
    """Measure detection recall, classification accuracy and speed on synthetic scenes."""
    from .evaluate import run_benchmark

    _setup_logging(False)
    cfg = _config(config, None, frame_skip)
    with console.status("Running benchmark scenes...") as status:
        bench_result = run_benchmark(
            cfg,
            scenes,
            seconds,
            cache,
            on_scene=lambda s: status.update(f"Scene {s + 1}/{scenes}..."),
        )
    total = bench_result.result
    frames, elapsed = bench_result.frames, bench_result.seconds

    classes = sorted({t for t, _ in total.confusion})
    predicted = sorted({p for _, p in total.confusion})
    table = Table(title="Confusion matrix (rows = truth)", header_style="bold cyan")
    table.add_column("")
    for p in predicted:
        table.add_column(p, justify="right")
    for t in classes:
        table.add_row(t, *(str(total.confusion.get((t, p), 0)) for p in predicted))
    console.print(table)
    console.print(
        f"Objects: {total.n_truth}  Recall: {total.recall:.1%}  "
        f"Classification accuracy: {total.accuracy:.1%}  False positives: {total.false_positives}"
    )
    console.print(f"Throughput: {frames / elapsed:.1f} frames/s ({frames} frames)")


@app.command()
def train(
    data: Annotated[
        list[Path] | None, typer.Option(help="Training CSVs (default: synthetic_train.csv)")
    ] = None,
    test: Annotated[
        list[Path] | None,
        typer.Option(help="Held-out CSVs (default: synthetic_test.csv, standard_test.csv)"),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Model file (default: from the config)")] = None,
    station: Annotated[
        bool,
        typer.Option(
            "--station/--no-station",
            help="Also learn from dashboard reviews and orbit-matched satellites",
        ),
    ] = True,
    config: ConfigOpt = None,
) -> None:
    """Train the learned classifier on labelled tracks and compare it with the rules.

    Build the datasets first with tools/build_training_set.py. Real labelled tracks from
    the station database (dashboard reviews, orbit-matched satellites) are added too.
    Results on the held-out sets are printed and saved next to the model
    (<model>_report.json); select the model with ``classifier: {method: model}``.
    """
    from .classify.dataset import arrays, load_csv, object_scores, station_rows
    from .classify.model import CLASSES, LearnedClassifier

    _setup_logging(False)
    cfg = load_config(config)
    folder = Path("data/datasets")
    train_files = data or [folder / "synthetic_train.csv"]
    test_files = test or [p for p in (folder / "synthetic_test.csv", folder / "standard_test.csv")]
    rows = [r for f in train_files for r in load_csv(f) if not int(r["star"])]
    real = station_rows(_store(cfg), cfg) if station else []
    if real:
        kinds = Counter(r["kind"] for r in real)
        console.print(f"Adding {len(real)} real tracks from the station database: {dict(kinds)}")
    rows += real
    X, y, groups, _ = arrays(rows)
    with console.status(f"Training on {len(rows)} tracks from {len(set(groups))} videos..."):
        info = {
            "trained_on": [str(f) for f in train_files],
            "tracks": len(rows),
            "real_tracks": len(real),
        }
        model = LearnedClassifier.train(X, y, groups, cfg.rules, info)

    report: dict[str, Any] = {"train": dict(info), "tests": {}}
    for f in test_files:
        if not f.exists():
            console.print(f"[yellow]Skipping missing test set {f}[/yellow]")
            continue
        all_rows = load_csv(f)
        tracks = [r for r in all_rows if not int(r["star"])]
        Xt, yt, _, rule_labels = arrays(tracks)
        scores = model.evaluate(Xt, yt, rule_labels)
        Xa, _, _, rule_all = arrays(all_rows)
        predicted = np.array(CLASSES)[model.probabilities(Xa).argmax(axis=1)]
        scores["objects"] = {
            "model": object_scores(all_rows, predicted),
            "rules": object_scores(all_rows, rule_all),
        }
        report["tests"][f.stem] = scores
        console.print(_training_table(f.stem, scores))

    path = out or Path(cfg.classifier.model_path)
    model.info["report"] = report
    model.save(path)
    report_path = path.with_name(path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"Model saved to [bold]{path}[/bold]; report in {report_path}")


def _training_table(name: str, scores: dict[str, Any]) -> Table:
    table = Table(title=f"Held-out: {name}", header_style="bold cyan")
    table.add_column("Measure")
    table.add_column("Rules", justify="right")
    table.add_column("Model", justify="right")
    m, r = scores["model"], scores["rules"]
    om, orr = scores["objects"]["model"], scores["objects"]["rules"]
    table.add_row("Objects found (recall)", f"{orr['recall']:.1%}", f"{om['recall']:.1%}")
    table.add_row("Objects classified right", f"{orr['accuracy']:.1%}", f"{om['accuracy']:.1%}")
    table.add_row("False alarms", str(orr["false_positives"]), str(om["false_positives"]))
    table.add_row("Track accuracy", f"{r['accuracy']:.1%}", f"{m['accuracy']:.1%}")
    table.add_row("Track macro-F1", f"{r['macro_f1']:.3f}", f"{m['macro_f1']:.3f}")
    for c in m["per_class"]:
        table.add_row(
            f"  F1 {c} (n={m['per_class'][c]['support']})",
            f"{r['per_class'][c]['f1']:.3f}",
            f"{m['per_class'][c]['f1']:.3f}",
        )
    table.add_row("Calibration error (ECE)", "-", f"{m['ece']:.3f}")
    table.add_row("Brier score", "-", f"{m['brier']:.3f}")
    return table


@app.command()
def calibrate(
    source: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Night-sky video or photo")
    ],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory")] = None,
    config: ConfigOpt = None,
    time: Annotated[
        datetime | None,
        typer.Option(
            "--time",
            help="UTC time of the recording start / photo (e.g. 2026-10-07T21:30:00)",
            formats=TIME_FORMATS,
        ),
    ] = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    fov: Annotated[float | None, typer.Option(help="Approx. horizontal field of view, deg")] = None,
    at: Annotated[float, typer.Option(help="Seconds into the video to calibrate at")] = 0.0,
    verbose: VerboseOpt = False,
) -> None:
    """Work out exactly where the camera points, from the stars in the image."""
    import cv2

    from .astro.calibrate import CalibrationError, load_sky_image
    from .astro.calibrate import calibrate as solve
    from .astro.catalog import build_database
    from .astro.overlay import draw_calibration

    _setup_logging(verbose)
    cfg = load_config(config)
    a = cfg.astro
    observer = _observer(cfg, lat, lon, elevation)
    if observer is None:
        console.print(
            "[yellow]No location given: solving sky coordinates only "
            "(add --lat/--lon for compass direction and altitude).[/yellow]"
        )
    when = None if time is None else (time if time.tzinfo else time.replace(tzinfo=UTC))

    data_dir = Path(a.data_dir)
    with console.status("Preparing star database (first run downloads and builds it)..."):
        db = build_database(a.database_min_fov, a.database_max_fov, a.database_max_mag, data_dir)
    image, frame_time = load_sky_image(source, when, at, a.calibration_seconds)
    try:
        with console.status("Recognising stars..."):
            cal = solve(
                image,
                frame_time,
                observer,
                fov_estimate=fov or a.fov_estimate,
                fov_max_error=a.fov_tolerance if (fov or a.fov_estimate) else None,
                database=db,
            )
    except CalibrationError as e:
        console.print(f"[red]Calibration failed:[/red] {e}")
        raise typer.Exit(1) from e

    out_dir = out or Path("data/calibrations") / f"{source.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cal.save(out_dir / "calibration.json")
    cv2.imwrite(
        str(out_dir / "calibration.png"),
        draw_calibration(np.clip(image, 0, 255).astype(np.uint8), cal),
    )

    s = cal.summary()
    table = Table(title="Camera calibration", show_header=False, title_style="bold cyan")
    table.add_column(style="cyan")
    table.add_column(justify="right")
    if observer is not None:
        table.add_row(
            "Pointing azimuth", f"{s['pointing_az']:.2f}°  ({_compass(s['pointing_az'])})"
        )
        table.add_row("Pointing altitude", f"{s['pointing_alt']:.2f}°")
        table.add_row("Image roll", f"{s['pointing_roll']:+.2f}°")
    table.add_row("Image centre RA / Dec", f"{s['center_ra']:.3f}° / {s['center_dec']:+.3f}°")
    table.add_row("Field of view", f"{s['fov_h']:.2f}° x {s['fov_v']:.2f}°")
    table.add_row("Pixel scale", f'{s["pixel_scale_arcsec"]:.1f}"/px')
    table.add_row("Lens distortion k1, k2", f"{s['distortion_k1']:+.4f}, {s['distortion_k2']:+.4f}")
    table.add_row("Stars matched", str(s["stars_matched"]))
    table.add_row("Residual", f'{s["rms_arcsec"]:.0f}" ({s["rms_px"]:.2f} px)')
    console.print(table)
    console.print(f"Saved calibration and overlay to [bold]{out_dir}[/bold]")


def _compass(az: float) -> str:
    points = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    return points[int((az % 360) / 22.5 + 0.5) % 16]


@app.command()
def showers(
    date: Annotated[
        datetime | None,
        typer.Option("--date", help="UTC date/time (default: now)", formats=TIME_FORMATS),
    ] = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    config: ConfigOpt = None,
    min_members: Annotated[
        int, typer.Option(help="Only well-observed showers (meteors in the IAU data)")
    ] = 1000,
) -> None:
    """Meteor showers active on a date, and (with --lat/--lon) where their radiants are."""
    from .astro.frames import icrs_to_enu_matrix, radec_to_vec, vec_to_altaz
    from .astro.showers import active_showers, solar_longitude

    cfg = load_config(config)
    when = date or datetime.now(UTC)
    when = when if when.tzinfo else when.replace(tzinfo=UTC)
    observer = _observer(cfg, lat, lon, None)
    lsun = solar_longitude(when)
    table = Table(
        title=f"Active meteor showers, {when:%Y-%m-%d %H:%M} UTC (solar longitude {lsun:.2f}°)",
        header_style="bold cyan",
    )
    for col in ("Code", "Shower", "Peak", "Radiant RA / Dec", "Speed"):
        table.add_column(col, justify="left" if col in ("Code", "Shower") else "right")
    if observer is not None:
        table.add_column("Radiant altitude", justify="right")
    rows = sorted(active_showers(when, min_members), key=lambda s: -s.members)
    for s in rows:
        ra, dec = s.radiant(lsun)
        days = ((s.lsun_peak - lsun + 180) % 360) - 180  # ~1 deg of solar longitude per day
        peak = (
            "today" if abs(days) < 1 else (f"in {days:.0f} d" if days > 0 else f"{-days:.0f} d ago")
        )
        row = [s.code, s.name, peak, f"{ra:6.1f}° / {dec:+5.1f}°", f"{s.v_inf:.0f} km/s"]
        if observer is not None:
            alt = float(
                vec_to_altaz(radec_to_vec(ra, dec) @ icrs_to_enu_matrix(when, observer).T)[0]
            )
            row.append(f"{alt:+.0f}°" + ("" if alt > 0 else " (below horizon)"))
        table.add_row(*row)
    console.print(table)
    console.print("[dim]Source: IAU Meteor Data Center (established showers).[/dim]")


@app.command()
def orbits(
    config: ConfigOpt = None,
    update: Annotated[bool, typer.Option("--update", help="Force a fresh download")] = False,
) -> None:
    """Download (or check) the satellite orbit catalogue used to name satellites."""
    from .astro.orbits import load_catalog as load_orbits
    from .astro.orbits import update_catalog

    _setup_logging(False)
    cfg = load_config(config)
    s = cfg.satellites
    data_dir = Path(cfg.astro.data_dir)
    with console.status("Contacting CelesTrak..."):
        paths = update_catalog(s.groups, data_dir, s.max_age_hours, force=update)
        cat = load_orbits(s.groups, data_dir, [Path(p) for p in s.extra_tle_files], download=False)
    ages = cat.epoch_age_days(datetime.now(UTC))
    console.print(f"{len(cat):,} satellites from {', '.join(str(p) for p in paths)}")
    console.print(
        f"Orbital elements are {float(np.median(ages)):.1f} days old (median); "
        "they stay accurate for about a week."
    )


aircraft_app = typer.Typer(
    help="Aircraft broadcasts (ADS-B): what is overhead now, or record them while filming.",
    no_args_is_help=True,
)
app.add_typer(aircraft_app, name="aircraft")
SourceOpt = Annotated[
    str | None,
    typer.Option("--source", help="opensky (free online service) or readsb (a local receiver)"),
]


def _aircraft_source(cfg: Config, observer: Observer | None, source: str | None) -> Any:
    from .aircraft import make_source

    if observer is None:
        console.print("[red]The location is needed (--lat/--lon or observer: in the config).")
        raise typer.Exit(1)
    return make_source(cfg.aircraft, observer, source or _default_source(cfg))


def _default_source(cfg: Config) -> str:
    return cfg.aircraft.source if cfg.aircraft.source != "none" else "opensky"


@aircraft_app.command("now")
def aircraft_now(
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    source: SourceOpt = None,
    config: ConfigOpt = None,
) -> None:
    """List the aircraft around you right now, nearest first, and where to look."""
    import numpy as np

    from .aircraft import enu_basis, geodetic_to_ecef
    from .astro.frames import vec_to_altaz

    cfg = _config(config, None, None)
    observer = _observer(cfg, lat, lon, elevation)
    src = _aircraft_source(cfg, observer, source)
    assert observer is not None
    with console.status(f"Asking {src.name}..."):
        states = src.fetch()
    origin = geodetic_to_ecef(observer.latitude, observer.longitude, observer.elevation_m)
    to_enu = enu_basis(observer.latitude, observer.longitude)
    rows = []
    for st in states:
        enu = (geodetic_to_ecef(st.lat, st.lon, st.alt_m) - origin) @ to_enu.T
        dist = float(np.linalg.norm(enu))
        if dist <= cfg.aircraft.radius_km * 1000:
            look_alt, look_az = vec_to_altaz(enu / dist)
            rows.append((dist, st, float(look_alt), float(look_az)))
    table = Table(title=f"{len(rows)} aircraft within {cfg.aircraft.radius_km:.0f} km")
    for col in ("Flight", "ICAO", "Distance", "Height", "Speed", "Look: alt", "az"):
        table.add_column(col, justify="left" if col in ("Flight", "ICAO") else "right")
    for dist, st, look_alt_deg, look_az_deg in sorted(rows, key=lambda r: r[0]):
        speed = "-" if st.speed_m_s is None else f"{st.speed_m_s * 3.6:.0f} km/h"
        table.add_row(
            st.callsign or "?",
            st.icao24.upper(),
            f"{dist / 1000:.0f} km",
            f"{st.alt_m:,.0f} m",
            speed,
            f"{look_alt_deg:.0f}°",
            f"{look_az_deg:.0f}°",
        )
    console.print(table)


@aircraft_app.command("record")
def aircraft_record(
    out: Annotated[Path, typer.Option("--out", "-o", help="Broadcast log to write (.jsonl)")],
    minutes: Annotated[float, typer.Option(help="How long to record")] = 30.0,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    source: SourceOpt = None,
    config: ConfigOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Record aircraft broadcasts while you film, for `skyseer analyze --aircraft`.

    Start it when you start filming (any camera works) and keep the computer online. With
    OpenSky and no account, 30 minutes uses ~180 of the 400 free requests a day.
    """
    import time

    from .aircraft import AircraftFeed, AircraftLog

    _setup_logging(verbose)
    cfg = _config(config, None, None)
    observer = _observer(cfg, lat, lon, elevation)
    src = _aircraft_source(cfg, observer, source)
    broadcasts = AircraftLog(path=out)
    feed = AircraftFeed(src, broadcasts, cfg.aircraft)
    end = time.time() + minutes * 60
    console.print(f"Recording broadcasts from {src.name} into {out} (Ctrl+C to stop)")
    try:
        while time.time() < end:
            feed.poll()
            st = feed.status()
            console.print(
                f"{datetime.now():%H:%M:%S}  {st['aircraft']} aircraft, "
                f"{len(broadcasts):,} positions, {st['requests_today']} requests"
                + (f"  [yellow]{st['error']}[/yellow]" if st["error"] else ""),
                highlight=False,
            )
            time.sleep(cfg.aircraft.poll_s)
    except KeyboardInterrupt:
        pass
    console.print(f"Saved {len(broadcasts):,} positions to {out}")


@app.command()
def version() -> None:
    """Print the SkySeer version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
