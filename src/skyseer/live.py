"""The live station: camera -> pipeline -> database, clips, alerts and the dashboard.

A background thread runs every frame through the pipeline. Each finished event gets its
clip, is stored in the database (so it appears in the dashboard and survives restarts),
is appended to ``events.jsonl`` in the night's folder (a crash-safe log) and is pushed to
every open dashboard over a WebSocket. A small annotated preview frame is kept for the
live video view. On stop, the folder gets events.json and run.json, like an analysed
recording, so it can be re-imported anywhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .config import Config
from .models import Event, Frame, ObjectClass
from .pipeline import Pipeline
from .render import CLASS_COLORS, _bgr
from .reports import summary
from .sources import FrameSource
from .storage import Store

if TYPE_CHECKING:
    from .aircraft import AircraftFeed, AircraftIdentifier
    from .astro.autocal import AutoCalibrator
    from .astro.frames import Observer
    from .astro.satellites import SatelliteIdentifier

log = logging.getLogger(__name__)

PREVIEW_WIDTH = 960
PREVIEW_FPS = 5.0
SHOWN_EVENTS_S = 6.0  # finished events stay drawn on the preview this long

_Queue = asyncio.Queue[dict[str, Any]]


class LiveStation:
    def __init__(
        self,
        source: FrameSource,
        config: Config,
        store: Store,
        camera: str,
        folder: str | Path,
        observer: Observer | None = None,
        autocal: AutoCalibrator | None = None,
        identifier: SatelliteIdentifier | None = None,
        clips: bool = True,
        on_event: Callable[[Event], None] | None = None,
        simulated: bool = False,
        aircraft: AircraftIdentifier | None = None,
        aircraft_feed: AircraftFeed | None = None,
    ):
        """``aircraft`` names planes from ADS-B broadcasts; ``aircraft_feed`` keeps its log
        up to date (it is asked only while something crosses the view, see
        :meth:`_crossing`). Events are checked once broadcasts after them have arrived."""
        self.source, self.config, self.store = source, config, store
        self.camera = camera
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.observer = observer
        self.autocal = autocal
        self.identifier = identifier
        self.clips = clips
        self.on_event = on_event
        self.simulated = simulated  # replaying a simulated video
        self.session_id: int | None = None
        self.started_at: datetime | None = None
        self.frames = 0
        self.error: str | None = None
        self.counts: dict[str, int] = {}
        self._fps = 0.0
        self._preview: bytes | None = None
        self._recent: deque[tuple[float, Event]] = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline: Pipeline | None = None
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, _Queue]] = []
        self._lock = threading.Lock()
        self.aircraft = aircraft
        self.aircraft_feed = aircraft_feed
        if aircraft_feed is not None and aircraft_feed.active is None:
            aircraft_feed.active = self._crossing
        self._last_crossing = -1e9  # time.monotonic() when something last crossed the view
        self._last_time: datetime | None = None  # timestamp of the latest frame
        self._awaiting: dict[int, Event] = {}  # events waiting for aircraft broadcasts
        self._timers: list[threading.Timer] = []
        self._aircraft_lock = threading.Lock()

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self.started_at = datetime.now(UTC)
        info = {
            "kind": "live",
            "source": self.camera,
            "fps": self.source.fps,
            "frame_size": [self.source.width, self.source.height],
            "simulated": self.simulated,
        }
        if self.observer is not None:
            info["observer"] = {
                "lat": self.observer.latitude,
                "lon": self.observer.longitude,
                "elevation_m": self.observer.elevation_m,
            }
        self.session_id = self.store.add_session(
            self.folder.name,
            "live",
            self.folder,
            self.camera,
            self.source.start_time.isoformat(),
            info,
        )
        self._thread = threading.Thread(target=self._run, name="live-station", daemon=True)
        self._thread.start()
        if self.aircraft_feed is not None:
            self.aircraft_feed.start()
        log.info("Live station started: camera %s, session %s", self.camera, self.session_id)

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def wait(self) -> None:
        """Block until the source ends (a replayed file) or the station is stopped."""
        if self._thread is not None:
            while self._thread.is_alive():
                self._thread.join(0.5)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------- dashboard API
    def status(self) -> dict[str, Any]:
        cal = self._pipeline.calibration if self._pipeline is not None else None
        pointing = None
        if cal is not None and cal.observer is not None:
            pointing = {k: round(v, 2) for k, v in cal.camera.pointing_altaz().items()}
        return {
            "running": self.running,
            "camera": self.camera,
            "fps": round(self._fps, 1),
            "camera_fps": self.source.fps,
            "frames": self.frames,
            "session_id": self.session_id,
            "started": self.started_at.isoformat() if self.started_at else None,
            "calibrated": cal is not None,
            "pointing": pointing,
            "observer": (
                {"lat": self.observer.latitude, "lon": self.observer.longitude}
                if self.observer is not None
                else None
            ),
            "counts": dict(self.counts),
            "error": self.error,
            "aircraft": None if self.aircraft_feed is None else self.aircraft_feed.status(),
        }

    def preview_jpeg(self) -> bytes | None:
        return self._preview

    async def updates(self) -> AsyncGenerator[dict[str, Any], None]:
        """Status now and every 2 s, plus every new event as it happens."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        entry = (asyncio.get_running_loop(), queue)
        with self._lock:
            self._subscribers.append(entry)
        try:
            yield {"type": "status", "live": self.status()}
            while True:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    yield {"type": "status", "live": self.status()}
        finally:
            with self._lock:
                self._subscribers.remove(entry)

    def _publish(self, message: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(_put, queue, message)

    # --------------------------------------------------------------- the night
    def _run(self) -> None:
        from .clips import ClipRecorder

        source, cfg = self.source, self.config
        pipeline = Pipeline(
            cfg,
            source.fps,
            (source.width, source.height),
            autocal=self.autocal,
            identifier=self.identifier,
            aircraft=self.aircraft,
            defer_aircraft=self.aircraft_feed is not None,
        )
        self._pipeline = pipeline
        recorder = ClipRecorder(self.folder, cfg.clips, source.fps) if self.clips else None
        window: deque[float] = deque()
        next_preview = 0.0
        try:
            for frame in source:
                if self._stop.is_set():
                    break
                if recorder is not None:
                    recorder.feed(frame)
                for event in pipeline.process(frame):
                    self._emit(event, recorder)
                if self.aircraft_feed is not None and _moving(pipeline.tracker.active_tracks):
                    self._last_crossing = time.monotonic()
                self.frames += 1
                self._last_time = frame.timestamp
                now = time.perf_counter()
                window.append(now)
                while window and now - window[0] > 2.0:
                    window.popleft()
                self._fps = (len(window) - 1) / max(window[-1] - window[0], 1e-6)
                if now >= next_preview:
                    self._preview = self._draw_preview(frame, pipeline)
                    next_preview = now + 1.0 / PREVIEW_FPS
            for event in pipeline.finish():
                self._emit(event, recorder)
        except Exception as exc:  # report it in the dashboard rather than dying silently
            log.exception("Live station stopped by an error")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if recorder is not None:
                recorder.close()
            source.close()
            if self.autocal is not None:
                self.autocal.close()
            self._check_waiting_aircraft()
            if self.aircraft_feed is not None:
                self.aircraft_feed.stop()
            self._finish(pipeline)
            self._publish({"type": "status", "live": self.status()})

    def _emit(self, event: Event, recorder: Any) -> None:
        written = recorder.save(event) if recorder is not None else None  # sets event.media
        report = event.to_dict()
        assert self.session_id is not None
        db_id = self.store.add_event(self.session_id, report)
        with open(self.folder / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report) + "\n")
        label = str(event.label)
        self.counts[label] = self.counts.get(label, 0) + 1
        if event.aircraft_pending:
            with self._aircraft_lock:
                self._awaiting[event.id] = event
            self._schedule_aircraft(event, 1)
        if event.label in (ObjectClass.JUNK, ObjectClass.STAR):
            return
        self._recent.append((time.perf_counter(), event))
        item = self._card(report, db_id, event.id)
        if written is None:
            self._publish({"type": "event", "event": item})
        else:
            # The clip and images are written in the background: tell dashboards once they
            # exist, so a new card never shows a missing image.
            written.add_done_callback(lambda job: self._publish_event(item, job))
        if self.on_event is not None:
            self.on_event(event)

    def _card(self, report: dict[str, Any], db_id: int, event_id: int) -> dict[str, Any]:
        """What dashboards show for an event (an event card), with media URLs."""
        item = summary({**report, "id": db_id, "session_id": self.session_id, "event_id": event_id})
        if item.get("media"):
            base = f"/media/{self.session_id}/"
            item["media"] = {
                k: base + v if k in ("clip", "thumbnail", "context", "clean") else v
                for k, v in item["media"].items()
            }
        return item

    # ------------------------------------------------------------ aircraft
    def _crossing(self) -> bool:
        """Is something crossing the view, or did it just finish? (The aircraft feed then
        asks for broadcasts, including those just after the pass.)"""
        cfg = self.config.aircraft
        return time.monotonic() - self._last_crossing < cfg.check_delay_s + 2 * cfg.poll_s

    def _schedule_aircraft(self, event: Event, attempt: int) -> None:
        delay = self.config.aircraft.check_delay_s
        timer = threading.Timer(delay, self._check_aircraft, (event.id, attempt))
        timer.daemon = True
        with self._aircraft_lock:
            self._timers = [t for t in self._timers if t.is_alive()] + [timer]
        timer.start()

    def _check_aircraft(self, event_id: int, attempt: int, final: bool = False) -> None:
        """Match an event against the broadcasts once they cover it (up to 3 tries; at the
        end of the night, with whatever has arrived)."""
        from .aircraft import apply_match

        assert self.aircraft is not None
        with self._aircraft_lock:
            event = self._awaiting.pop(event_id, None)
        if event is None:
            return  # already checked
        if not final and attempt < 3 and not self.aircraft.covers(event.sky):
            with self._aircraft_lock:
                self._awaiting[event_id] = event
            self._schedule_aircraft(event, attempt + 1)
            return
        before = str(event.label)
        try:
            apply_match(event, self.aircraft.identify(event.sky))
        except Exception:  # never let one check stop the station
            log.exception("Aircraft check of event %d failed", event.id)
            return
        finally:
            event.aircraft_pending = False
        if event.aircraft is None and str(event.label) == before:
            report = event.to_dict()  # record that the broadcasts were checked
            assert self.session_id is not None
            self.store.add_event(self.session_id, report)
            return
        log.info("Event %d identified as aircraft %s", event.id, event.aircraft.short_label())
        self._update(event, before)

    def _update(self, event: Event, label_before: str) -> None:
        """Store and announce an event whose identification changed after it was shown."""
        report = event.to_dict()
        assert self.session_id is not None
        db_id = self.store.add_event(self.session_id, report)  # replaces the stored event
        with open(self.folder / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report) + "\n")
        label = str(event.label)
        if label != label_before:
            self.counts[label_before] = max(0, self.counts.get(label_before, 1) - 1)
            self.counts[label] = self.counts.get(label, 0) + 1
        self._publish({"type": "event", "event": self._card(report, db_id, event.id)})

    def _check_waiting_aircraft(self) -> None:
        """End of the night: check every event still waiting, with what has arrived."""
        with self._aircraft_lock:
            timers, waiting = list(self._timers), list(self._awaiting)
        for t in timers:
            t.cancel()
        for event_id in waiting:
            self._check_aircraft(event_id, 3, final=True)

    def _publish_event(self, item: dict[str, Any], job: Future[None]) -> None:
        if job.exception() is not None:
            item = {**item, "media": None}
        self._publish({"type": "event", "event": item})

    def _finish(self, pipeline: Pipeline) -> None:
        """Close the session: events.json and run.json make the folder self-contained."""
        if self.session_id is None:
            return
        # The session ends at its last frame's time: for a replay that is the video's clock,
        # like its start (the wall clock gave replays lengths of days).
        ended = self._last_time or datetime.now(UTC)
        fps = self.source.fps
        meta = {
            "ended": ended.isoformat(),
            "disturbances": [
                [round(a / fps, 2), round(b / fps, 2)] for a, b in pipeline.disturbances
            ],
            "discarded_tracks": pipeline.discarded_tracks,
            "frames": self.frames,
        }
        cal = pipeline.calibration
        if cal is not None:
            meta["calibration"] = cal.summary()
            if cal.observer is not None:
                meta["pointing"] = cal.camera.pointing_altaz()
        self.store.finish_session(self.session_id, ended.isoformat(), meta)
        session = self.store.session(self.session_id) or {}
        events = self.store.events(session_id=self.session_id, limit=1_000_000)
        reports = sorted(
            (
                {k: v for k, v in e.items() if k not in ("session_id", "review", "event_id")}
                | {"id": e["event_id"]}
                for e in events
            ),
            key=lambda e: e["start_time"],
        )
        (self.folder / "events.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
        info = {**session.get("meta", {}), "started": session.get("started")}
        (self.folder / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        log.info("Live session %s closed: %d frames", self.session_id, self.frames)

    def _draw_preview(self, frame: Frame, pipeline: Pipeline) -> bytes | None:
        """The current view, small, with tracked objects bracketed and recent events
        drawn in their class colour."""
        image = frame.image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        scale = min(1.0, PREVIEW_WIDTH / image.shape[1])
        view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        view = cv2.convertScaleAbs(view, alpha=1.8, beta=6)  # brighten the dark sky a little
        accent = (255, 208, 57)
        for track in pipeline.tracker.active_tracks:
            if len(track.detections) < 3:
                continue
            pts = (track.positions()[-60:] * scale).astype(np.int32)
            cv2.polylines(view, [pts], False, (120, 100, 40), 1, cv2.LINE_AA)
            x, y = pts[-1]
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                cx, cy = x + dx * 9, y + dy * 9
                cv2.line(view, (cx, cy), (cx - dx * 5, cy), accent, 1, cv2.LINE_AA)
                cv2.line(view, (cx, cy), (cx, cy - dy * 5), accent, 1, cv2.LINE_AA)
        now = time.perf_counter()
        while self._recent and now - self._recent[0][0] > SHOWN_EVENTS_S:
            self._recent.popleft()
        for _, event in self._recent:
            color = _bgr(CLASS_COLORS[event.label])
            pts = (event.track.positions() * scale).astype(np.int32)
            cv2.polylines(view, [pts], False, color, 2, cv2.LINE_AA)
            name = event.identification.name.strip() if event.identification else str(event.label)
            x, y = pts[-1]
            cv2.putText(
                view,
                name,
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        stamp = f"{frame.timestamp:%Y-%m-%d %H:%M:%S} UTC  {self._fps:4.1f} fps"
        cv2.putText(
            view,
            stamp,
            (10, view.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes() if ok else None


def _moving(tracks: list[Any]) -> bool:
    """Is any tracked object really moving across the view (not a twinkling star)?"""
    for t in tracks:
        d = t.detections
        if len(d) >= 5 and np.hypot(d[-1].x - d[0].x, d[-1].y - d[0].y) > 10:
            return True
    return False


def _put(queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
    if queue.full():  # a slow dashboard: drop its oldest message rather than block
        queue.get_nowait()
    queue.put_nowait(message)
