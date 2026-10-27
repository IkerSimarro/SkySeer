"""Analyses of uploaded videos, run one at a time in the background.

Each analysis runs in its own low-priority process (worker.py), so the computer stays
usable and the dashboard responsive. One at a time keeps a Raspberry Pi responsive too.
Jobs are kept in the station database, so the list survives a restart; an analysis the
server was running when it stopped can be resumed from its last save (about every five
minutes of video).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..analysis import can_resume
from ..config import Config
from ..storage import Store
from .worker import SPEEDS

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")


class Jobs:
    def __init__(self, config: Config, store: Store):
        self.config = config
        self.store = store
        self._jobs: dict[str, dict[str, Any]] = {}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        for job in store.jobs():  # analyses from before a restart
            if job["state"] in ACTIVE:
                job.update(state="stopped", message="Stopped when the server closed")
            job["resumable"] = _resumable(job)
            self._jobs[job["id"]] = job
            store.save_job(job)

    def submit(
        self,
        video: Path,
        start_time: datetime | None = None,
        lat: float | None = None,
        lon: float | None = None,
        elevation: float = 0.0,
        sky: bool = True,
        speed: str = "full",
    ) -> dict[str, Any]:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        job_id = uuid.uuid4().hex[:12]
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
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
            "start_progress": 0.0,  # where it began: above 0 when resumed
            "speed": speed,
            "resumable": False,
            "out_dir": str(Path(self.config.storage.runs_dir) / f"{video.stem}_{stamp}"),
            "params": {
                "video": str(video),
                "start_time": start_time.isoformat() if start_time else None,
                "lat": lat,
                "lon": lon,
                "elevation": elevation,
                "sky": sky,
            },
        }
        with self._lock:
            self._jobs[job_id] = job
        self.store.save_job(job)
        self._executor.submit(self._run, job_id, False)
        return self.get(job_id) or job

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _public(job) if job else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [_public(j) for j in self._jobs.values()]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def stop(self, job_id: str) -> dict[str, Any] | None:
        """Stop a queued or running analysis; it keeps its last save, to resume later."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] not in ACTIVE:
                return _public(job) if job else None
            job.update(state="stopped", message="Stopped")
            proc = self._procs.get(job_id)
        if proc is not None:
            proc.terminate()
            proc.wait()
        self._finish_stop(job_id)
        return self.get(job_id)

    def resume(self, job_id: str) -> dict[str, Any] | None:
        """Carry on a stopped analysis from its last save (or start it, if it never did)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.get("resumable"):
                return None
            job.update(state="queued", message="Waiting to resume", error=None, resumable=False)
        self._save(job_id)
        self._executor.submit(self._run, job_id, True)
        return self.get(job_id)

    def remove(self, job_id: str) -> bool:
        """Take a finished, failed or stopped analysis off the list (files are kept)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in ACTIVE:
                return False
            del self._jobs[job_id]
        self.store.delete_job(job_id)
        return True

    def close(self) -> None:
        """Stop running analyses (each keeps its last save) when the server shuts down."""
        for job_id in [j["id"] for j in self.all() if j["state"] in ACTIVE]:
            self.stop(job_id)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ internals
    def _update(self, job_id: str, save: bool = True, **changes: Any) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return  # removed meanwhile
            self._jobs[job_id].update(changes)
        if save:
            self._save(job_id)

    def _save(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            job = dict(job) if job else None
        if job is not None:
            self.store.save_job(job)

    def _finish_stop(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["resumable"] = _resumable(job)
            if job["state"] == "stopped" and job["progress"] > 0:
                job["message"] = f"Stopped at {100 * job['progress']:.0f}%: it can be resumed"
        self._save(job_id)

    def _run(self, job_id: str, resume: bool) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] != "queued":
                return  # stopped or removed while waiting
            out_dir = Path(job["out_dir"])
            spec = {
                **job["params"],
                "out_dir": str(out_dir),
                "speed": job["speed"],
                "resume": resume and can_resume(out_dir),  # else it never got going: start
                "database": str(self.store.path),
                "config": self.config.model_dump(mode="json"),
                "parent_pid": os.getpid(),
            }
            job.update(
                state="running",
                message="Starting",
                started=datetime.now(UTC).isoformat(),
                start_progress=job["progress"] if resume else 0.0,
            )
        self._save(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_file = out_dir / "job.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "skyseer.api.worker", str(spec_file)],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        with self._lock:
            self._procs[job_id] = proc
        outcome: dict[str, Any] = {}
        saved = 0.0
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "progress" in msg:
                save = time.monotonic() - saved > 10  # the database only every 10 s
                saved = time.monotonic() if save else saved
                self._update(job_id, save=save, progress=msg["progress"])
            elif "message" in msg:
                self._update(job_id, message=msg["message"])
            else:
                outcome = msg
        code = proc.wait()
        with self._lock:
            self._procs.pop(job_id, None)
            job = self._jobs.get(job_id)
            if job is None or job["state"] == "stopped":
                return  # stop() records it
        if outcome.get("done"):
            found = outcome.get("found", 0)
            self._update(
                job_id,
                state="done",
                progress=1.0,
                session_id=outcome.get("session_id"),
                message=f"Done: {found} object{'s' if found != 1 else ''} found",
            )
            spec_file.unlink(missing_ok=True)
            return
        error = outcome.get("error") or f"The analysis process ended unexpectedly (code {code})"
        log.error("Analysis job %s failed: %s", job_id, error)
        self._update(job_id, save=False, state="failed", error=error)
        self._finish_stop(job_id)


def _resumable(job: dict[str, Any]) -> bool:
    """A stopped analysis can carry on (from its last save, or from the start if it never
    began); a failed one only if it had saved some progress."""
    return job["state"] == "stopped" or (
        job["state"] == "failed" and can_resume(Path(job["out_dir"]))
    )


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """What the dashboard sees (no file paths or parameters)."""
    return {k: v for k, v in job.items() if k not in ("out_dir", "params")}
