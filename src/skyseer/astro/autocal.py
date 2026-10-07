"""Keep a live camera calibrated without ever blocking the detection loop."""

from __future__ import annotations

import contextlib
import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import AstroConfig
from ..models import Frame
from .calibrate import Calibration, CalibrationError, calibrate, stack_frames
from .catalog import build_database
from .frames import Observer

log = logging.getLogger(__name__)


class AutoCalibrator:
    """Buffers a few seconds of frames and re-solves the camera in a background thread.

    * First solve as soon as the buffer is full (retrying every ``retry_s`` if the sky is
      cloudy or not dark yet).
    * Afterwards, re-checks every ``interval_s``, starting from the previous solution so it
      is fast; a moved camera falls back to a full solve automatically.
    * The latest good solution is saved to ``cache_path`` and reused after a restart.
    """

    def __init__(
        self,
        config: AstroConfig,
        observer: Observer | None,
        fps: float,
        interval_s: float = 1800.0,
        retry_s: float = 120.0,
        cache_path: Path | None = None,
    ):
        self.config = config
        self.observer = observer
        self.interval_s = interval_s
        self.retry_s = retry_s
        self.cache_path = cache_path
        n_frames = 15
        self._every = max(1, round(config.calibration_seconds * fps / n_frames))
        self._frames: deque[tuple[np.ndarray, datetime]] = deque(maxlen=n_frames)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calibrate")
        self._future: Future[Calibration] | None = None
        self._last_attempt: datetime | None = None
        self._last_success: datetime | None = None
        self.calibration: Calibration | None = None
        if cache_path is not None and cache_path.exists():
            try:
                cached = Calibration.load(cache_path)
                if cached.observer == observer:
                    self.calibration = cached
                    log.info("Loaded previous calibration from %s", cache_path)
            except (OSError, ValueError, KeyError) as e:
                log.warning("Ignoring unreadable calibration cache %s: %s", cache_path, e)

    def feed(self, frame: Frame) -> None:
        self._collect_result()
        if frame.index % self._every:
            return
        gray = (
            frame.image if frame.image.ndim == 2 else cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        )
        self._frames.append((gray, frame.timestamp))
        if (
            self._future is None
            and len(self._frames) == self._frames.maxlen
            and self._due(frame.timestamp)
        ):
            self._start(frame.timestamp)

    def _due(self, now: datetime) -> bool:
        if self._last_attempt is None:
            return True
        since_attempt = (now - self._last_attempt).total_seconds()
        if self._last_success is None:
            return since_attempt >= self.retry_s
        return (now - self._last_success).total_seconds() >= self.interval_s and (
            since_attempt >= self.retry_s
        )

    def _start(self, now: datetime) -> None:
        images = [f for f, _ in self._frames]
        mid_time = self._frames[len(self._frames) // 2][1]
        self._last_attempt = now
        a = self.config
        hint = self.calibration

        def job() -> Calibration:
            # Database build (first run only) also happens off the detection thread.
            db = build_database(
                a.database_min_fov, a.database_max_fov, a.database_max_mag, Path(a.data_dir)
            )
            return calibrate(
                stack_frames(images),
                mid_time,
                self.observer,
                fov_estimate=a.fov_estimate,
                fov_max_error=a.fov_tolerance if a.fov_estimate else None,
                database=db,
                hint=hint,
            )

        self._future = self._executor.submit(job)

    def _collect_result(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        try:
            cal = future.result()
        except CalibrationError as e:
            log.info("Calibration attempt failed: %s", e)
            return
        except Exception:  # never let a calibration bug stop detection
            log.exception("Calibration crashed")
            return
        self.calibration = cal
        self._last_success = cal.time
        log.info('Camera calibrated: %d stars, %.0f" RMS', cal.n_stars, cal.rms_arcsec)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cal.save(self.cache_path)

    def wait(self, timeout: float | None = None) -> None:
        """Block until an in-flight calibration finishes (used at shutdown and in tests)."""
        if self._future is not None:
            # Errors are collected and logged by _collect_result below.
            with contextlib.suppress(Exception):
                self._future.result(timeout=timeout)
        self._collect_result()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
