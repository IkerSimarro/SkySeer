"""The station's web API, and the dashboard it serves.

A headless camera station is used from a phone or laptop, so everything SkySeer knows
is available over HTTP: sessions (nights or recordings), events with their clips and
images, night reports, exports, human reviews and station health. The built dashboard
(``web/``, compiled into ``api/static``) is served from the same address.

There is no login: by default the server listens only on this computer (127.0.0.1).
Listening on the home network is a deliberate choice (``skyseer serve --host 0.0.0.0``).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

from .. import __version__
from ..config import Config
from ..reports import REAL, events_csv, night_report, summary
from ..storage import LABELS, Store
from .jobs import Jobs

STATIC_DIR = Path(__file__).parent / "static"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"}


class Station(Protocol):
    """A running live camera (see live.py); the API only reads from it."""

    def status(self) -> dict[str, Any]: ...

    def preview_jpeg(self) -> bytes | None: ...

    def updates(self) -> AsyncIterator[dict[str, Any]]:
        """Messages for the dashboard: new events, health, calibration changes."""
        ...


class Review(BaseModel):
    label: str | None  # Satellite, Meteor, Plane, Junk, Star; None clears the review


def _media(e: dict[str, Any]) -> dict[str, Any] | None:
    """Turn an event's relative clip/image paths into URLs."""
    media = e.get("media")
    if not media:
        return None
    base = f"/media/{e['session_id']}/"
    return {k: base + v if k in ("clip", "thumbnail", "context") else v for k, v in media.items()}


def _with_urls(e: dict[str, Any]) -> dict[str, Any]:
    return {**e, "media": _media(e)}


def health(config: Config, started: float, station: Station | None) -> dict[str, Any]:
    import psutil

    temperature = None
    sensors = getattr(psutil, "sensors_temperatures", None)  # Linux (the Pi) only
    if sensors is not None:
        for entries in (sensors() or {}).values():
            if entries:
                temperature = round(float(entries[0].current), 1)
                break
    data = Path(config.storage.database).resolve().parent
    data.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(data)
    return {
        "version": __version__,
        "uptime_s": round(time.time() - started),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "temperature_c": temperature,
        "disk_free_gb": round(disk.free / 1e9, 1),
        "disk_used_percent": round(100 * disk.used / disk.total, 1),
        "live": station.status() if station is not None else None,
    }


def create_app(
    config: Config,
    store: Store,
    station: Station | None = None,
    static_dir: Path | None = STATIC_DIR,
) -> FastAPI:
    app = FastAPI(title="SkySeer", version=__version__, docs_url="/api/docs")
    started = time.time()
    jobs = Jobs(config, store)
    uploads = Path(config.storage.database).parent / "uploads"

    def session_or_404(session_id: int) -> dict[str, Any]:
        s = store.session(session_id)
        if s is None:
            raise HTTPException(404, f"No session {session_id}")
        return s

    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        return health(config, started, station)

    @app.get("/api/sessions")
    def get_sessions() -> list[dict[str, Any]]:
        return store.sessions()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: int) -> dict[str, Any]:
        return session_or_404(session_id)

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: int) -> dict[str, bool]:
        session_or_404(session_id)
        store.delete_session(session_id)  # records only: the folder stays on disk
        return {"deleted": True}

    @app.get("/api/sessions/{session_id}/report")
    def get_report(session_id: int) -> dict[str, Any]:
        s = session_or_404(session_id)
        return night_report(s, store.events(session_id=session_id, limit=1_000_000))

    @app.get("/api/sessions/{session_id}/export.csv")
    def export_csv(session_id: int, include_junk: bool = False) -> Response:
        s = session_or_404(session_id)
        events = _session_events(store, session_id, include_junk)
        return PlainTextResponse(
            events_csv(events),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{s["name"]}.csv"'},
        )

    @app.get("/api/sessions/{session_id}/export.json")
    def export_json(session_id: int, include_junk: bool = False) -> Response:
        import json

        s = session_or_404(session_id)
        events = _session_events(store, session_id, include_junk)
        return Response(
            json.dumps({"session": s, "events": events}, indent=1),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{s["name"]}.json"'},
        )

    @app.get("/api/events")
    def get_events(
        session_id: int | None = None,
        label: Annotated[list[str] | None, Query()] = None,
        reviewed: bool | None = None,
        since: str | None = None,
        until: str | None = None,
        include_junk: bool = False,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        labels = label or (None if include_junk else list(REAL))
        events = store.events(session_id, labels, reviewed, since, until, limit, offset)
        simulated = {x["id"]: bool(x["meta"].get("simulated")) for x in store.sessions()}
        return [
            {**_with_urls(summary(e)), "simulated": simulated.get(e["session_id"], False)}
            for e in events
        ]

    @app.get("/api/events/{event_id}")
    def get_event(event_id: int) -> dict[str, Any]:
        e = store.event(event_id)
        if e is None:
            raise HTTPException(404, f"No event {event_id}")
        s = store.session(e["session_id"]) or {"meta": {}}
        return {**_with_urls(e), "simulated": bool(s["meta"].get("simulated"))}

    @app.put("/api/events/{event_id}/review")
    def put_review(event_id: int, review: Review) -> dict[str, Any]:
        if review.label is not None and review.label not in LABELS:
            raise HTTPException(400, f"label must be one of {LABELS} or null")
        if not store.review(event_id, review.label):
            raise HTTPException(404, f"No event {event_id}")
        return {"id": event_id, "review": review.label}

    @app.post("/api/analyze")
    async def post_analyze(
        file: UploadFile,
        start_time: Annotated[str | None, Form()] = None,
        lat: Annotated[float | None, Form()] = None,
        lon: Annotated[float | None, Form()] = None,
        elevation: Annotated[float, Form()] = 0.0,
        sky: Annotated[bool, Form()] = True,
    ) -> dict[str, Any]:
        """Upload a recording to analyse in the background (see /api/jobs)."""
        import re
        from datetime import UTC, datetime

        name = Path(file.filename or "video").name
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"Not a video file: {name}")
        when = None
        if start_time:
            try:
                when = datetime.fromisoformat(start_time)
            except ValueError as exc:
                raise HTTPException(400, f"start_time must be ISO 8601: {start_time}") from exc
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)  # times without a zone are UTC
        uploads.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[-80:]
        dest = uploads / f"{datetime.now():%Y%m%d_%H%M%S}_{safe}"
        with open(dest, "wb") as fh:  # in chunks: night recordings can be gigabytes
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        return jobs.submit(dest, when, lat, lon, elevation, sky)

    @app.get("/api/jobs")
    def get_jobs() -> list[dict[str, Any]]:
        return jobs.all()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No job {job_id}")
        return job

    @app.get("/media/{session_id}/{path:path}")
    def get_media(session_id: int, path: str) -> FileResponse:
        folder = Path(session_or_404(session_id)["folder"]).resolve()
        file = (folder / path).resolve()
        if not file.is_relative_to(folder) or not file.is_file():  # no ../ escapes
            raise HTTPException(404, "Not found")
        return FileResponse(file)

    @app.get("/api/live/preview.mjpg")
    def live_preview() -> StreamingResponse:
        if station is None:
            raise HTTPException(404, "No live camera running")
        mime = "multipart/x-mixed-replace; boundary=frame"
        return StreamingResponse(_mjpeg(station), media_type=mime)

    @app.websocket("/ws/live")
    async def live_updates(ws: WebSocket) -> None:
        await ws.accept()
        if station is None:
            await ws.send_json({"type": "status", "live": None})
            await ws.close()
            return
        try:
            async for message in station.updates():
                await ws.send_json(message)
        except WebSocketDisconnect:
            pass

    if static_dir is not None and (static_dir / "index.html").exists():
        _serve_dashboard(app, static_dir)
    return app


def _session_events(store: Store, session_id: int, include_junk: bool) -> list[dict[str, Any]]:
    labels = None if include_junk else list(REAL)
    events = store.events(session_id=session_id, labels=labels, limit=1_000_000)
    return sorted(({**e, "id": e["event_id"]} for e in events), key=lambda e: e["start_time"])


async def _mjpeg(station: Station) -> Any:
    import asyncio

    while True:
        jpeg = station.preview_jpeg()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.2)


def _serve_dashboard(app: FastAPI, static_dir: Path) -> None:
    """Serve the built dashboard; unknown paths get index.html (client-side routes).

    index.html is always re-checked, so an updated station shows the new dashboard at
    once; the build's assets have a content hash in their names, so they can be cached
    for good.
    """
    root = static_dir.resolve()
    fresh = {"Cache-Control": "no-cache"}
    forever = {"Cache-Control": "public, max-age=31536000, immutable"}

    @app.get("/{path:path}", include_in_schema=False)
    def dashboard(path: str) -> FileResponse:
        if path.startswith(("api/", "media/", "ws/")):
            raise HTTPException(404, "Not found")
        file = (root / path).resolve()
        if path and file.is_relative_to(root) and file.is_file():
            return FileResponse(file, headers=forever if path.startswith("assets/") else fresh)
        return FileResponse(root / "index.html", headers=fresh)
