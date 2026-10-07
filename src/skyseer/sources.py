"""Frame sources: anything that yields timestamped frames (video files, cameras)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import cv2

from .models import Frame

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    fps: float
    width: int
    height: int
    start_time: datetime

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class FileSource:
    """Reads a video file. Timestamps are ``start_time + index / fps``.

    ``start_time`` should be the real recording start (UTC) when known, because the
    astronomy layer needs absolute time. Without it, the file's modification time is used.
    """

    def __init__(self, path: str | Path, start_time: datetime | None = None):
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_time is None:
            start_time = datetime.fromtimestamp(self.path.stat().st_mtime, tz=UTC) - timedelta(
                seconds=self.frame_count / self.fps
            )
            log.info("No start time given; assuming %s from file mtime", start_time.isoformat())
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)  # naive times are taken as UTC
        self.start_time = start_time.astimezone(UTC)

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield Frame(image, index, self.start_time + timedelta(seconds=index / self.fps))
            index += 1

    def close(self) -> None:
        self._cap.release()


class CameraSource:
    """Live OpenCV camera (USB webcam index or RTSP/HTTP URL). Timestamps are wall-clock."""

    def __init__(self, device: int | str = 0, width: int | None = None, height: int | None = None):
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open camera: {device}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start_time = datetime.now(UTC)

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                log.warning("Camera read failed; stopping")
                break
            yield Frame(image, index, datetime.now(UTC))
            index += 1

    def close(self) -> None:
        self._cap.release()
