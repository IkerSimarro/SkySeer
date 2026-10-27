"""One dashboard analysis, run in its own process: ``python -m skyseer.api.worker JOB.json``.

A separate process can run at low priority, so whatever else the computer is doing (a
game, the browser) gets the processor first and the analysis takes only what is left.
It also lets the server stop an analysis at once. Progress goes to stdout as one JSON
object per line, which the server (jobs.py) reads; log messages go to stderr.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# The speed choices: "full" detail, or a "quick" look (detection on a half-size picture:
# 2.6x faster on 1080p, but it found only 10 of 24 faint satellites on a bright sky).
SPEEDS = ("full", "quick")


def threads_for(cores: int | None = None) -> int:
    """Half the cores, leaving the rest free: on an 8-thread PC all 8 were no faster than
    4 (28.7 against 30.3 frames/s on 1080p video, measured)."""
    cores = cores or os.cpu_count() or 1
    return max(1, cores // 2)


def lower_priority() -> None:
    """Below-normal priority (Windows) or niceness 10 (Linux, the Pi)."""
    import psutil

    me = psutil.Process()
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)


def _exit_with_parent(parent: int) -> None:
    """If the server dies, stop too, rather than finish unseen: the last save stays,
    so the analysis can be resumed when the server is back."""
    import psutil

    while True:
        time.sleep(2)
        if not psutil.pid_exists(parent):
            os._exit(3)


def emit(**message: Any) -> None:
    print(json.dumps(message), flush=True)


def run(spec: dict[str, Any]) -> int:
    import cv2

    from .. import sources
    from ..analysis import analyze_video
    from ..config import Config
    from ..storage import Store

    speed = spec.get("speed", "full")
    lower_priority()
    threads = threads_for()
    cv2.setNumThreads(threads)
    sources.DECODE_THREADS = threads
    cfg = Config.model_validate(spec["config"])
    if speed == "quick":  # half-size detection: faster, but faint satellites can be missed
        cfg = cfg.model_copy(
            update={"detector": cfg.detector.model_copy(update={"process_scale": 0.5})}
        )
    observer = None
    if spec.get("lat") is not None and spec.get("lon") is not None:
        from ..astro.frames import Observer

        observer = Observer(spec["lat"], spec["lon"], spec.get("elevation") or 0.0)
    start = datetime.fromisoformat(spec["start_time"]) if spec.get("start_time") else None
    last = -1.0

    def progress(done: int, total: int) -> None:
        nonlocal last
        fraction = round(done / total, 3) if total else 0.0
        if fraction != last:  # at most ~1000 lines for a whole video
            last = fraction
            emit(progress=fraction)

    result = analyze_video(
        Path(spec["video"]),
        cfg,
        Path(spec["out_dir"]),
        start,
        observer,
        sky=spec.get("sky", True),
        store=Store(spec["database"]),
        progress=progress,
        notify=lambda m: emit(message=m),
        resume=spec.get("resume", False),
    )
    emit(done=True, session_id=result.session_id, found=len(result.earlier) + len(result.reported))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    spec = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    if spec.get("parent_pid"):
        threading.Thread(target=_exit_with_parent, args=(spec["parent_pid"],), daemon=True).start()
    try:
        return run(spec)
    except Exception as exc:  # reported on the dashboard
        log.exception("Analysis of %s failed", spec.get("video"))
        emit(error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
