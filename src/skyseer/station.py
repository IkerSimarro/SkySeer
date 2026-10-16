"""The unattended station: observe every night, sleep by day, never fill the disk.

``Station`` runs the live camera (``live.LiveStation``) while the Sun is below
``station.dark_sun_altitude`` and stops it at dawn. After each night it tidies the disk:
clips and images of old live nights are deleted (the event records stay in the
database), oldest first, when they are older than ``keep_clips_days`` or free space falls
below ``min_free_gb``. Analysed recordings are never touched. If the camera fails (for
example it was unplugged), it retries a minute later.

The dashboard talks to it exactly as to a single live night (status, preview, updates).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config
from .storage import Store

if TYPE_CHECKING:
    from .astro.frames import Observer
    from .gps import GpsMonitor
    from .live import LiveStation

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Station:
    def __init__(
        self,
        config: Config,
        store: Store,
        make_night: Callable[[], LiveStation],
        observer: Observer | None,
        clock: Clock = _utcnow,
        gps: GpsMonitor | None = None,
    ):
        self.config = config
        self.store = store
        self.make_night = make_night
        self.fixed_observer = observer  # given by the user: always wins over the GPS
        self.gps = gps
        self.clock = clock
        self.night: LiveStation | None = None
        self.next_change: datetime | None = None
        self.last_error: str | None = None
        self.nights = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def observer(self) -> Observer | None:
        """Where the station is: as given, else from the GPS (or its last saved fix)."""
        if self.fixed_observer is not None:
            return self.fixed_observer
        return self.gps.location() if self.gps is not None else None

    def clock_warning(self) -> str | None:
        """A message when the system clock disagrees with GPS time."""
        fix = self.gps.fix if self.gps is not None else None
        offset = fix.clock_offset_s() if fix is not None else None
        limit = self.config.gps.max_clock_error_s
        if offset is None or abs(offset) <= limit:
            return None
        return (
            f"The system clock is {offset:+.1f} s off GPS time: satellite names may fail. "
            "Let chrony set the clock from the GPS (deploy/pi/README.md)."
        )

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.gps is not None:
            self.gps.start()
        self._thread = threading.Thread(target=self._run, name="station", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self.gps is not None:
            self.gps.stop()
        if self.night is not None:
            self.night.stop()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_dark(self) -> bool:
        cfg = self.config.station
        if cfg.always_on or self.observer is None:
            return True  # without a location there is no sunset to wait for
        from .astro.sun import is_dark

        return is_dark(self.clock(), self.observer, cfg.dark_sun_altitude)

    def sun_altitude(self) -> float | None:
        if self.observer is None:
            return None
        from .astro.sun import sun_altitude

        return round(sun_altitude(self.clock(), self.observer), 2)

    # ------------------------------------------------------------ dashboard API
    def status(self) -> dict[str, Any]:
        night = self.night
        base = (
            night.status()
            if night is not None
            else {
                "running": False,
                "camera": self.config.station.camera,
                "fps": 0.0,
                "frames": 0,
                "session_id": None,
                "calibrated": False,
            }
        )
        return {
            **base,
            "mode": "observing" if night is not None and night.running else "waiting",
            "sun_altitude": self.sun_altitude(),
            "dark_below": self.config.station.dark_sun_altitude,
            "next_change": self.next_change.isoformat() if self.next_change else None,
            "nights": self.nights,
            "station_error": self.last_error,
            "location": (
                {"lat": self.observer.latitude, "lon": self.observer.longitude}
                if self.observer is not None
                else None
            ),
            "gps": self.gps.status() if self.gps is not None else None,
            "clock_warning": self.clock_warning(),
        }

    def preview_jpeg(self) -> bytes | None:
        night = self.night
        return night.preview_jpeg() if night is not None and night.running else None

    async def updates(self) -> AsyncGenerator[dict[str, Any], None]:
        """Follow whichever night is running; status every few seconds in between."""
        while True:
            night = self.night
            if night is None or not night.running:
                yield {"type": "status", "live": self.status()}
                await asyncio.sleep(3.0)
                continue
            stream = night.updates()
            try:
                async for message in stream:
                    if message["type"] == "status":
                        message = {"type": "status", "live": self.status()}
                    yield message
                    if self.night is not night or not night.running:
                        break  # the night ended: switch to the next one
            finally:
                await stream.aclose()

    # ------------------------------------------------------------------- nights
    def _run(self) -> None:
        cfg = self.config.station
        while not self._stop.is_set():
            if not self.is_dark():
                self._schedule()
                self._stop.wait(cfg.check_interval_s)
                continue
            self._observe()
            if not self._stop.is_set():
                self.cleanup()

    def _schedule(self) -> None:
        if self.observer is None:
            return
        from .astro.sun import next_change

        now = self.clock()
        if self.next_change is None or self.next_change <= now:
            limit = self.config.station.dark_sun_altitude
            self.next_change = next_change(now, self.observer, limit)

    def _observe(self) -> None:
        cfg = self.config.station
        try:
            night = self.make_night()
        except Exception as exc:  # camera missing or busy: try again shortly
            self.last_error = f"Could not start the camera: {exc}"
            log.warning(self.last_error)
            self._stop.wait(cfg.retry_s)
            return
        self.last_error = None
        self.night = night
        self.nights += 1
        self._schedule()
        night.start()
        log.info("Night %d started (sun %.1f°)", self.nights, self.sun_altitude() or 0.0)
        while not self._stop.is_set() and night.running:
            if not self.is_dark():
                log.info("Dawn: stopping the camera")
                break
            self._stop.wait(cfg.check_interval_s)
        night.stop()
        if night.status().get("error"):
            self.last_error = night.status()["error"]
            self._stop.wait(cfg.retry_s)

    # -------------------------------------------------------------- housekeeping
    def cleanup(self) -> list[str]:
        """Delete clips of old live nights (oldest first); returns the sessions tidied."""
        cfg = self.config.station
        live_dir = Path(self.config.storage.live_dir)
        live_dir.mkdir(parents=True, exist_ok=True)
        cutoff = self.clock() - timedelta(days=cfg.keep_clips_days)
        sessions = [
            s
            for s in self.store.sessions()
            if s["kind"] == "live"
            and not s["meta"].get("clips_deleted")
            and (Path(s["folder"]) / "clips").is_dir()
            and (self.night is None or s["id"] != self.night.session_id)
        ]
        sessions.sort(key=lambda s: s["started"] or "")
        tidied = []
        for s in sessions:
            started = datetime.fromisoformat(s["started"]) if s["started"] else None
            old = started is not None and started < cutoff
            low = shutil.disk_usage(live_dir).free / 1e9 < cfg.min_free_gb
            if not (old or low):
                continue
            clips = Path(s["folder"]) / "clips"
            freed = sum(f.stat().st_size for f in clips.rglob("*") if f.is_file())
            shutil.rmtree(clips, ignore_errors=True)
            when = self.clock().isoformat()
            self.store.finish_session(s["id"], s["ended"] or when, {"clips_deleted": when})
            run = Path(s["folder"]) / "run.json"
            if run.exists():
                info = json.loads(run.read_text(encoding="utf-8"))
                run.write_text(json.dumps({**info, "clips_deleted": when}, indent=2), "utf-8")
            reason = "older than the limit" if old else "low disk space"
            log.info("Deleted clips of %s (%.0f MB, %s)", s["name"], freed / 1e6, reason)
            tidied.append(s["name"])
        return tidied
