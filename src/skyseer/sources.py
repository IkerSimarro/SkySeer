"""Frame sources: anything that yields timestamped frames (video files, cameras)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import av
import cv2
import numpy as np

from .models import Frame

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    fps: float
    width: int
    height: int
    start_time: datetime

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class VideoFile:
    """A video file read with PyAV (FFmpeg): decoded on several threads, and straight to
    greyscale when colour isn't needed (on 1080p H.264, 2.7 ms a frame against 10.5 ms
    for OpenCV's reader plus conversion). Frames are numbered from 0 as in the file and
    turned upright like OpenCV does (phones store portrait video rotated)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self._container = av.open(str(self.path))
            self._stream = self._container.streams.video[0]
        except (av.error.FFmpegError, IndexError) as exc:
            raise ValueError(f"Could not open video file: {self.path}") from exc
        s = self._stream
        s.thread_type = "AUTO"  # decode several frames at once
        self.fps = float(s.average_rate or s.guessed_rate or 30.0)
        seconds = (
            float(s.duration * s.time_base)
            if s.duration
            else float(self._container.duration or 0) / 1e6
        )
        self.frame_count = int(s.frames) if s.frames else int(round(seconds * self.fps))
        first = next(self._container.decode(s), None)
        if first is None:
            raise ValueError(f"No frames in video file: {self.path}")
        self._pts0 = first.pts or 0  # frame numbers count from the first frame's time
        self._turns = int(round((getattr(first, "rotation", 0) or 0) / 90)) % 4
        self.width, self.height = first.width, first.height
        if self._turns % 2:
            self.width, self.height = self.height, self.width
        self._container.seek(0, stream=s)

    def frames(
        self, first: int = 0, last: int | None = None, gray: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """(index, image) from frame ``first`` up to (not including) ``last``; images
        are BGR, or 2-D greyscale with ``gray``."""
        s = self._stream
        tb = s.time_base
        # To the keyframe at or before ``first``, then decode forward.
        self._container.seek(self._pts0 + int(first / self.fps / tb), stream=s)
        fmt = "gray" if gray else "bgr24"
        prev = -1
        for frame in self._container.decode(s):
            if frame.pts is None:
                index = prev + 1
            else:  # from the frame's own time, never repeating or going backwards
                index = max(prev + 1, round(float((frame.pts - self._pts0) * tb) * self.fps))
            prev = index
            if index < first:
                continue
            if last is not None and index >= last:
                break
            image = frame.to_ndarray(format=fmt)
            if self._turns:
                image = np.ascontiguousarray(np.rot90(image, self._turns))
            yield index, image

    def close(self) -> None:
        self._container.close()


class FileSource:
    """Reads a video file. Timestamps are ``start_time + index / fps``.

    ``start_time`` should be the real recording start (UTC) when known, because the
    astronomy layer needs absolute time. Without it, the file's modification time is used.
    """

    def __init__(
        self,
        path: str | Path,
        start_time: datetime | None = None,
        from_s: float = 0.0,
        to_s: float | None = None,
        gray: bool = False,
    ):
        """``from_s``/``to_s`` limit processing to part of the video. Frame indices and
        timestamps stay those of the full video. ``gray`` yields 2-D greyscale frames
        (faster; all the analysis needs unless clips are kept in colour)."""
        self.path = Path(path)
        self.from_s, self.to_s = from_s, to_s
        self.gray = gray
        self._video = VideoFile(self.path)
        self.fps = self._video.fps
        self.width, self.height = self._video.width, self._video.height
        self.frame_count = self._video.frame_count
        if start_time is None:
            start_time = datetime.fromtimestamp(self.path.stat().st_mtime, tz=UTC) - timedelta(
                seconds=self.frame_count / self.fps
            )
            log.info("No start time given; assuming %s from file mtime", start_time.isoformat())
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)  # naive times are taken as UTC
        self.start_time = start_time.astimezone(UTC)

    def __iter__(self) -> Iterator[Frame]:
        first = int(self.from_s * self.fps)
        last = None if self.to_s is None else int(self.to_s * self.fps)
        for index, image in self._video.frames(first, last, self.gray):
            yield Frame(image, index, self.start_time + timedelta(seconds=index / self.fps))

    def close(self) -> None:
        self._video.close()


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


class ReplaySource:
    """Plays a video file as if it were a live camera: paced in real time (``speed`` times
    faster), so the live station and dashboard can be run and demonstrated without a
    camera. Timestamps start at ``start_time`` (the recording's real start, so satellites
    can be named) or, if not given, now.
    """

    def __init__(
        self,
        path: str | Path,
        start_time: datetime | None = None,
        speed: float = 1.0,
        loop: bool = False,
    ):
        self.path = Path(path)
        self.speed, self.loop = speed, loop
        self._video = VideoFile(self.path)
        self.fps = self._video.fps
        self.width, self.height = self._video.width, self._video.height
        start = start_time or datetime.now(UTC)
        self.start_time = (start if start.tzinfo else start.replace(tzinfo=UTC)).astimezone(UTC)

    def __iter__(self) -> Iterator[Frame]:
        import time

        index = 0
        began = time.perf_counter()
        while True:
            for _, image in self._video.frames():
                if self.speed > 0:  # wait until this frame's moment comes
                    due = began + index / (self.fps * self.speed)
                    delay = due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                yield Frame(image, index, self.start_time + timedelta(seconds=index / self.fps))
                index += 1
            if not self.loop:
                break

    def close(self) -> None:
        self._video.close()
