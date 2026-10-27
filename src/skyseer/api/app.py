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
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel

from .. import __version__
from ..clips import clean_path, write_clean_clip
from ..config import Config
from ..reports import REAL, events_csv, night_report, summary
from ..storage import LABELS, Store
from .jobs import Jobs
from .worker import SPEEDS

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


class Rename(BaseModel):
    title: str | None  # a name for the session; None or "" restores its folder name


def _media(e: dict[str, Any]) -> dict[str, Any] | None:
    """Turn an event's relative clip/image paths into URLs."""
    media = e.get("media")
    if not media:
        return None
    base = f"/media/{e['session_id']}/"
    urls = ("clip", "thumbnail", "context", "clean")
    return {k: base + v if k in urls else v for k, v in media.items()}


def _with_urls(e: dict[str, Any]) -> dict[str, Any]:
    return {**e, "media": _media(e)}


def _source_video(session: dict[str, Any]) -> Path | None:
    """The analysed recording's original video, if it is still where it was analysed."""
    source = session.get("source") or session["meta"].get("source")
    if session["kind"] != "file" or not source:
        return None
    video = Path(source)
    return video if video.is_file() else None


def _clean_clip(e: dict[str, Any], session: dict[str, Any], config: Config) -> dict[str, Any]:
    """Can the event's clip be shown without markings? ``ready`` (with its URL),
    ``on_request`` (made from the original recording when asked) or ``unavailable``
    (with the reason)."""
    media = e.get("media") or {}
    if not media.get("clip") or session["meta"].get("clips_deleted"):
        return {"clean_status": "unavailable", "clean_reason": "This event has no clip."}
    rel = clean_path(media["clip"])
    if (Path(session["folder"]) / rel).is_file():
        url = f"/media/{e['session_id']}/{rel}"
        return {"clean": url, "clean_status": "ready", "clean_reason": None}
    if session["kind"] != "file":
        kept = " and ".join(f"{c.lower()}s" for c in config.clips.clean_copy) or "no events"
        why = f"Live nights aren't recorded, so a clip without markings is saved only for {kept}."
        return {"clean_status": "unavailable", "clean_reason": why}
    if _source_video(session) is None:
        why = "The original video is no longer where it was analysed, so this can't be made."
        return {"clean_status": "unavailable", "clean_reason": why}
    return {"clean_status": "on_request", "clean_reason": None}


def _flagged(e: dict[str, Any], session_meta: dict[str, Any]) -> dict[str, Any]:
    """Mark simulated data, and drop media whose clips were deleted to save space."""
    out = {**e, "simulated": bool(session_meta.get("simulated"))}
    if session_meta.get("clips_deleted"):
        out["media"] = None
    return out


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
    jobs = Jobs(config, store)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.close()  # running analyses stop at their last save, to resume later

    app = FastAPI(title="SkySeer", version=__version__, docs_url="/api/docs", lifespan=lifespan)
    started = time.time()
    clean_lock = threading.Lock()  # one clean clip made at a time (and never twice)
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
        """Every session, each with up to 3 of its most striking events (``highlights``)."""
        best = store.highlights()
        out = []
        for s in store.sessions():
            picks = [] if s["meta"].get("clips_deleted") else best.get(s["id"], [])
            base = f"/media/{s['id']}/"
            s["highlights"] = [{**h, "thumbnail": base + h["thumbnail"]} for h in picks]
            out.append(s)
        return out

    data_dirs = [Path(config.storage.runs_dir), Path(config.storage.live_dir), uploads]

    def own_file(path: str | None) -> Path | None:
        """``path`` if it exists inside SkySeer's own data folders (never anything else)."""
        if not path:
            return None
        p = Path(path).resolve()
        inside = any(p.is_relative_to(d.resolve()) and p != d.resolve() for d in data_dirs)
        return p if inside and p.exists() else None

    def files_of(s: dict[str, Any]) -> dict[str, Any]:
        folder = own_file(s["folder"])
        upload = own_file(s.get("source")) if s["kind"] == "file" else None
        if upload is not None and not upload.is_relative_to(uploads.resolve()):
            upload = None  # only videos uploaded through the dashboard
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) if folder else 0
        return {
            "folder": str(folder) if folder else None,
            "folder_mb": round(size / 1e6, 1),
            "upload": str(upload) if upload else None,
            "upload_mb": round(upload.stat().st_size / 1e6, 1) if upload else 0.0,
        }

    @app.get("/api/sessions/{session_id}/files")
    def get_session_files(session_id: int) -> dict[str, Any]:
        """What removing the session with its files would delete (SkySeer's own only)."""
        return files_of(session_or_404(session_id))

    @app.patch("/api/sessions/{session_id}")
    def rename_session(session_id: int, body: Rename) -> dict[str, Any]:
        session_or_404(session_id)
        title = (body.title or "").strip() or None
        if title is not None and len(title) > 80:
            raise HTTPException(400, "A session's name can be at most 80 characters")
        store.rename_session(session_id, title)
        return session_or_404(session_id)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: int) -> dict[str, Any]:
        return session_or_404(session_id)

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: int, files: bool = False) -> dict[str, Any]:
        """Remove a session (it won't be imported again). With ``files`` also delete its
        analysis folder (clips, reports) and its uploaded video, both only inside
        SkySeer's own data folders."""
        s = session_or_404(session_id)
        live = station.status() if station is not None else {}
        if live.get("running") and live.get("session_id") == session_id:
            raise HTTPException(409, "The live station is recording this session right now")
        freed = 0.0
        if files:
            found = files_of(s)
            if found["folder"]:
                shutil.rmtree(found["folder"], ignore_errors=True)
                freed += found["folder_mb"]
            if found["upload"]:
                Path(found["upload"]).unlink(missing_ok=True)
                freed += found["upload_mb"]
        store.delete_session(session_id)
        return {"deleted": True, "freed_mb": round(freed, 1)}

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
        flags = {x["id"]: x["meta"] for x in store.sessions()}
        return [_flagged(_with_urls(summary(e)), flags.get(e["session_id"], {})) for e in events]

    @app.get("/api/events/{event_id}")
    def get_event(event_id: int) -> dict[str, Any]:
        e = store.event(event_id)
        if e is None:
            raise HTTPException(404, f"No event {event_id}")
        s = session_or_404(e["session_id"])
        out = _flagged(_with_urls(e), s["meta"])
        if out["media"]:
            out["media"] = {**out["media"], **_clean_clip(e, s, config)}
        return out

    @app.post("/api/events/{event_id}/clean-clip")
    def make_clean_clip(event_id: int) -> dict[str, Any]:
        """Make the event's clip without markings (from the original recording) if it
        doesn't exist yet. Returns the event's media, now with the ``clean`` URL."""
        e = store.event(event_id)
        if e is None:
            raise HTTPException(404, f"No event {event_id}")
        s = session_or_404(e["session_id"])
        state = _clean_clip(e, s, config)
        if state["clean_status"] == "unavailable":
            raise HTTPException(409, state["clean_reason"])
        if state["clean_status"] == "on_request":
            video = _source_video(s)
            assert video is not None  # checked by _clean_clip
            with clean_lock:
                try:
                    write_clean_clip(e, video, Path(s["folder"]), config.clips)
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
        media: dict[str, Any] = get_event(event_id)["media"]
        return media

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
        speed: Annotated[str, Form()] = "full",
    ) -> dict[str, Any]:
        """Upload a recording to analyse in the background (see /api/jobs). ``speed`` is
        full (detail) or quick (a half-size picture: faster, may miss faint satellites).
        Both run at low priority on half the cores, so the computer stays usable."""
        import re
        from datetime import UTC, datetime

        name = Path(file.filename or "video").name
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"Not a video file: {name}")
        if speed not in SPEEDS:
            raise HTTPException(400, f"speed must be one of {SPEEDS}")
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
        return jobs.submit(dest, when, lat, lon, elevation, sky, speed)

    @app.get("/api/jobs")
    def get_jobs() -> list[dict[str, Any]]:
        return jobs.all()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No job {job_id}")
        return job

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        """Stop an analysis; it keeps its last save (about every 5 min of video)."""
        job = jobs.stop(job_id)
        if job is None:
            raise HTTPException(404, f"No job {job_id}")
        return job

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict[str, Any]:
        job = jobs.resume(job_id)
        if job is None:
            raise HTTPException(409, "Only a stopped analysis can be resumed")
        return job

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, bool]:
        """Take an analysis off the list (its files and session stay)."""
        if not jobs.remove(job_id):
            raise HTTPException(409, "Only a finished, failed or stopped analysis can be removed")
        return {"removed": True}

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
    elif static_dir is not None:

        @app.get("/", include_in_schema=False)
        def dashboard_missing() -> HTMLResponse:
            return HTMLResponse(NO_DASHBOARD)

    return app


NO_DASHBOARD = """<!doctype html><meta charset="utf-8"><title>SkySeer</title>
<body style="font-family:system-ui;background:#070b14;color:#e6edf6;max-width:40em;margin:4em auto">
<h1>SkySeer is running, but the dashboard hasn't been built</h1>
<p>The dashboard's web page is compiled from <code>web/</code> (it needs Node.js):</p>
<pre style="background:#111a2b;padding:1em">cd web
npm ci
npm run build</pre>
<p>Then reload this page. The API works already:
<a style="color:#5ec8ff" href="/api/docs">/api/docs</a>.</p>
</body>"""


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
