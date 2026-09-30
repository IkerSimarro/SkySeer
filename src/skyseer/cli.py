"""Command-line interface: ``skyseer analyze``, ``skyseer live``, ``skyseer synth``."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated

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
        writer = csv.writer(fh)
        writer.writerow(base + feature_keys)
        for r in rows:
            writer.writerow([r[k] for k in base] + [r["features"][k] for k in feature_keys])


def _summary(events: list[Event]) -> Table:
    table = Table(title=f"{len(events)} events", header_style="bold cyan")
    for col in ("#", "Class", "Conf.", "Start (UTC)", "Duration", "Speed px/f", "Linearity"):
        table.add_column(col, justify="right" if col not in ("Class",) else "left")
    for e in events:
        table.add_row(
            str(e.id),
            str(e.label),
            f"{e.confidence:.2f}",
            e.track.start_time.strftime("%H:%M:%S"),
            f"{e.track.duration:.1f}s",
            f"{e.features['avg_speed']:.2f}",
            f"{e.features['linearity']:.2f}",
        )
    return table


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory")] = None,
    config: ConfigOpt = None,
    start_time: Annotated[
        datetime | None, typer.Option(help="Recording start, UTC (e.g. 2026-09-30T21:15:00)")
    ] = None,
    sensitivity: Annotated[int | None, typer.Option(min=1, max=10)] = None,
    frame_skip: Annotated[int | None, typer.Option(min=1)] = None,
    video_out: Annotated[
        bool, typer.Option("--video/--no-video", help="Render annotated video")
    ] = True,
    include_junk: Annotated[bool, typer.Option(help="Also report Junk and Star events")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Analyse a recorded video and write events.json, events.csv and an annotated video."""
    _setup_logging(verbose)
    cfg = _config(config, sensitivity, frame_skip)
    out_dir = out or Path("data/runs") / f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    source = FileSource(video, start_time)

    with Progress(
        TextColumn("[cyan]Analysing"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=source.frame_count or None)
        events = list(run(source, cfg, lambda i: progress.update(task, completed=i)))
        progress.update(task, completed=source.frame_count)

    reported = events if include_junk else [e for e in events if e.label not in ("Junk", "Star")]
    write_reports(reported, out_dir)
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
    try:
        for e in run(source, cfg):
            if e.label not in ("Junk", "Star"):
                console.print(
                    f"[bold]{e.track.start_time:%H:%M:%S}[/bold] #{e.id} {e.label} "
                    f"({e.confidence:.2f}, {e.track.duration:.1f}s)"
                )
    except KeyboardInterrupt:
        console.print("Stopped.")


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
def version() -> None:
    """Print the SkySeer version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
