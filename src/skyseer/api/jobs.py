"""Analyses of uploaded videos, run one at a time in the background.

One at a time keeps a Raspberry Pi responsive (an analysis uses a whole CPU core), and
each job reports its progress and status so the dashboard can show them.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Config
from ..storage import Store

log = logging.getLogger(__name__)


class Jobs:
    def __init__(self, config: Config, store: Store):
        self.config = config
        self.store = store
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")

    def submit(
        self,
        video: Path,
        start_time: datetime | None = None,
        lat: float | None = None,
        lon: float | None = None,
        elevation: float = 0.0,
        sky: bool = True,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "video": video.name,
            "state": "queued",
            "progress": 0.0,
            "message": "Waiting for the previous analysis to finish",
            "session_id": None,
            "error": None,
            "created": datetime.now(UTC).isoformat(),
            "started": None,  # when the analysis itself began (for the time-left estimate)
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, video, start_time, lat, lon, elevation, sky)
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                (dict(j) for j in self._jobs.values()), key=lambda j: j["created"], reverse=True
            )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(
        self,
        job_id: str,
        video: Path,
        start_time: datetime | None,
        lat: float | None,
        lon: float | None,
        elevation: float,
        sky: bool,
    ) -> None:
        from ..analysis import analyze_video

        observer = None
        if lat is not None and lon is not None:
            from ..astro.frames import Observer

            observer = Observer(lat, lon, elevation)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        out_dir = Path(self.config.storage.runs_dir) / f"{video.stem}_{stamp}"
        self._update(
            job_id, state="running", message="Starting", started=datetime.now(UTC).isoformat()
        )
        try:
            result = analyze_video(
                video,
                self.config,
                out_dir,
                start_time,
                observer,
                sky=sky,
                store=self.store,
                progress=lambda i, n: self._update(job_id, progress=round(i / n, 3) if n else 0),
                notify=lambda m: self._update(job_id, message=m),
            )
        except Exception as exc:  # shown to the user on the dashboard
            log.exception("Analysis of %s failed", video)
            self._update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
            return
        found = sum(1 for e in result.reported)
        self._update(
            job_id,
            state="done",
            progress=1.0,
            session_id=result.session_id,
            message=f"Done: {found} object{'s' if found != 1 else ''} found",
        )
