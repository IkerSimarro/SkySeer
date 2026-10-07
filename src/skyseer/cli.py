"""Command-line interface: ``skyseer analyze | calibrate | live | synth | bench``."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .models import Event
from .pipeline import run
from .render import render_annotated_video
from .sources import CameraSource, FileSource

if TYPE_CHECKING:  # the astronomy stack is imported lazily to keep the CLI quick to start
    from .astro.calibrate import Calibration
    from .astro.frames import Observer
    from .astro.satellites import SatelliteIdentifier

app = typer.Typer(
    help="SkySeer: satellite and meteor detection for night-sky video.", no_args_is_help=True
)
console = Console()

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


def write_reports(events: list[Event], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(
        json.dumps([e.to_dict() for e in events], indent=2), encoding="utf-8"
    )
    with open(out_dir / "events.csv", "w", newline="", encoding="utf-8") as fh:
        rows = [e.to_dict() for e in events]
        feature_keys = list(rows[0]["features"]) if rows else []
        base = [
            "id",
            "label",
            "confidence",
            "start_time",
            "end_time",
            "duration_s",
            "first_frame",
            "last_frame",
        ]
        sky_cols = [
            "angular_speed_deg_s",
            "arc_deg",
            "start_alt",
            "start_az",
            "end_alt",
            "end_az",
            "satellite",
            "norad_id",
            "clock_offset_s",
        ]
        writer = csv.writer(fh)
        writer.writerow(base + sky_cols + feature_keys)
        for r in rows:
            sky = r.get("sky") or {}
            ident = r.get("identification") or {}
            start, end = sky.get("start", {}), sky.get("end", {})
            extra = [
                sky.get("angular_speed_deg_s", ""),
                sky.get("arc_deg", ""),
                start.get("alt", ""),
                start.get("az", ""),
                end.get("alt", ""),
                end.get("az", ""),
                ident.get("name", "").strip(),
                ident.get("norad_id", ""),
                ident.get("clock_offset_s", ""),
            ]
            writer.writerow([r[k] for k in base] + extra + [r["features"][k] for k in feature_keys])


def _summary(events: list[Event]) -> Table:
    table = Table(title=f"{len(events)} events", header_style="bold cyan")
    with_sky = any(e.sky is not None for e in events)
    with_id = any(e.identification_checked for e in events)
    cols = ["#", "Class", "Conf.", "Start (UTC)", "Duration"]
    cols += ["Speed °/s", "Arc °"] if with_sky else ["Speed px/f", "Linearity"]
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
        if with_id:
            m = e.identification
            row.append(
                m.short_label()
                if m is not None
                else ("[dim]no catalogued match[/dim]" if e.identification_checked else "")
            )
        table.add_row(*row)
    return table


LatOpt = Annotated[float | None, typer.Option("--lat", help="Observer latitude, degrees N")]
LonOpt = Annotated[float | None, typer.Option("--lon", help="Observer longitude, degrees E")]
ElevOpt = Annotated[float | None, typer.Option("--elevation", help="Observer elevation, m")]


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


def _calibrate_recording(
    video: Path, cfg: Config, observer: Observer | None, start_time: datetime | None
) -> Calibration | None:
    """Calibrate from the first seconds of a recording; None (with a warning) on failure."""
    from .astro.calibrate import CalibrationError, calibrate, load_sky_image
    from .astro.catalog import build_database

    a = cfg.astro
    try:
        with console.status("Preparing star database (first run downloads and builds it)..."):
            db = build_database(
                a.database_min_fov, a.database_max_fov, a.database_max_mag, Path(a.data_dir)
            )
        with console.status("Recognising stars to calibrate the camera..."):
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
        console.print(
            f"[yellow]Camera calibration failed ({e}); continuing without sky coordinates.[/yellow]"
        )
        return None
    s = cal.summary()
    where = (
        f"az {s['pointing_az']:.1f}°, alt {s['pointing_alt']:.1f}°"
        if observer is not None
        else f"RA {s['center_ra']:.1f}°, Dec {s['center_dec']:+.1f}°"
    )
    console.print(
        f"[green]Camera calibrated[/green] from {cal.n_stars} stars: pointing {where}, "
        f"field {s['fov_h']:.1f}° x {s['fov_v']:.1f}°"
    )
    return cal


TleOpt = Annotated[
    list[Path] | None,
    typer.Option("--tle", help="Extra TLE file(s), e.g. historical orbits for an old video"),
]
ClockOpt = Annotated[
    float | None,
    typer.Option("--clock-tolerance", help="How many seconds the camera clock may be off"),
]


def _identifier(
    cfg: Config, observer: Observer | None, tle: list[Path] | None, clock: float | None
) -> SatelliteIdentifier | None:
    """Satellite identifier from the orbit catalogue, or None if it can't be used."""
    if observer is None or not cfg.satellites.enabled:
        if cfg.satellites.enabled:
            console.print(
                "[yellow]No location given: satellites will not be named "
                "(add --lat/--lon).[/yellow]"
            )
        return None
    from .astro.orbits import load_catalog
    from .astro.satellites import SatelliteIdentifier

    s = cfg.satellites
    if clock is not None:
        s = s.model_copy(update={"max_clock_offset_s": clock})
    extra = [Path(p) for p in s.extra_tle_files] + list(tle or [])
    try:
        with console.status("Loading satellite orbits (downloads from CelesTrak if stale)..."):
            orbits = load_catalog(s.groups, Path(cfg.astro.data_dir), extra, s.max_age_hours)
    except RuntimeError as e:
        console.print(
            f"[yellow]Satellite orbits unavailable ({e}); satellites will not be named.[/yellow]"
        )
        return None
    console.print(f"Loaded orbits of {len(orbits):,} satellites")
    return SatelliteIdentifier(orbits, observer, s)


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory")] = None,
    config: ConfigOpt = None,
    start_time: Annotated[
        datetime | None, typer.Option(help="Recording start, UTC (e.g. 2026-09-30T21:15:00)")
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
    sensitivity: Annotated[int | None, typer.Option(min=1, max=10)] = None,
    frame_skip: Annotated[int | None, typer.Option(min=1)] = None,
    video_out: Annotated[
        bool, typer.Option("--video/--no-video", help="Render annotated video")
    ] = True,
    include_junk: Annotated[bool, typer.Option(help="Also report Junk and Star events")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Analyse a recorded video: detect, calibrate from the stars, name satellites.

    Writes events.json, events.csv, calibration.json and an annotated video.
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

    cal = None
    if calibration is not None:
        from .astro.calibrate import Calibration

        cal = Calibration.load(calibration)
    elif sky:
        cal = _calibrate_recording(video, cfg, observer, when)
    identifier = _identifier(cfg, cal.observer, tle, clock_tolerance) if cal is not None else None

    source = FileSource(video, when)
    with Progress(
        TextColumn("[cyan]Analysing"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=source.frame_count or None)
        events = list(
            run(
                source,
                cfg,
                lambda i: progress.update(task, completed=i),
                calibration=cal,
                identifier=identifier,
            )
        )
        progress.update(task, completed=source.frame_count)

    reported = events if include_junk else [e for e in events if e.label not in ("Junk", "Star")]
    write_reports(reported, out_dir)
    if cal is not None:
        cal.save(out_dir / "calibration.json")
    if video_out and reported:
        with console.status("Rendering annotated video..."):
            render_annotated_video(video, reported, out_dir / "annotated.mp4")

    counts = Counter(str(e.label) for e in events)
    console.print(_summary(reported))
    console.print("All tracks by class: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    console.print(f"Results written to [bold]{out_dir}[/bold]")


@app.command()
def live(
    camera: Annotated[str, typer.Option(help="Camera index or stream URL")] = "0",
    config: ConfigOpt = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    elevation: ElevOpt = None,
    sky: Annotated[
        bool, typer.Option("--sky/--no-sky", help="Keep the camera calibrated from the stars")
    ] = True,
    tle: TleOpt = None,
    clock_tolerance: ClockOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run detection on a live camera and print events as they happen (Ctrl+C to stop)."""
    _setup_logging(verbose)
    cfg = load_config(config)
    device: int | str = int(camera) if camera.isdigit() else camera
    source = CameraSource(device)
    console.print(
        f"Watching camera {camera} at {source.width}x{source.height}, {source.fps:.0f} fps"
    )
    autocal = identifier = None
    if sky:
        from .astro.autocal import AutoCalibrator

        observer = _observer(cfg, lat, lon, elevation)
        autocal = AutoCalibrator(
            cfg.astro,
            observer,
            source.fps,
            cache_path=Path(cfg.astro.data_dir) / "live_calibration.json",
        )
        identifier = _identifier(cfg, observer, tle, clock_tolerance)
    try:
        for e in run(source, cfg, autocal=autocal, identifier=identifier):
            if e.label not in ("Junk", "Star"):
                where = f", {e.sky.angular_speed:.2f}°/s" if e.sky is not None else ""
                name = f" = {e.identification.short_label()}" if e.identification else ""
                console.print(
                    f"[bold]{e.track.start_time:%H:%M:%S}[/bold] #{e.id} {e.label}{name} "
                    f"({e.confidence:.2f}, {e.track.duration:.1f}s{where})"
                )
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        if autocal is not None:
            autocal.close()


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
    import time

    from .evaluate import EvalResult, evaluate
    from .synth import GroundTruth, SceneSpec, generate

    _setup_logging(False)
    cfg = _config(config, None, frame_skip)
    total = EvalResult(0, 0, 0, 0)
    frames = 0
    elapsed = 0.0
    for seed in range(scenes):
        path = cache / f"scene_{seed:03d}_{int(seconds)}s.mp4"
        if path.exists() and path.with_suffix(".json").exists():
            truth = GroundTruth.load(path.with_suffix(".json"))
        else:
            with console.status(f"Generating scene {seed}..."):
                truth = generate(SceneSpec(seconds=seconds, seed=1000 + seed), path)
        source = FileSource(path)
        frames += source.frame_count
        t0 = time.perf_counter()
        events = list(run(source, cfg))
        elapsed += time.perf_counter() - t0
        r = evaluate(events, truth)
        total.n_truth += r.n_truth
        total.detected += r.detected
        total.correct += r.correct
        total.false_positives += r.false_positives
        total.confusion.update(r.confusion)

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
def calibrate(
    source: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Night-sky video or photo")
    ],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory")] = None,
    config: ConfigOpt = None,
    time: Annotated[
        datetime | None,
        typer.Option(
            "--time", help="UTC time of the recording start / photo (e.g. 2026-10-07T21:30:00)"
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


@app.command()
def version() -> None:
    """Print the SkySeer version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
