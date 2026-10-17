"""One short clip and two still images per event, for review and the dashboard.

The recorder is fed every frame, live or from a file, and keeps a memory-bounded buffer
of recent frames. When the pipeline reports an event it writes, in ``<run>/clips/``:

- ``0012.mp4``: a zoomed clip of the object's path, from a moment before it appears to a
  moment after, with brackets on the object and its trail drawn behind it. Long events
  become time-lapses (``media["speedup"]`` says how much faster than real time).
- ``0012.jpg``: the *trail image*: every frame of the event layered into one picture, so
  a satellite shows as a smooth line, a meteor as a streak and an aircraft as a dotted
  line. It is the event's thumbnail.
- ``0012_context.jpg``: the whole view with the path marked, to show where it happened.

Clips are H.264 MP4, which every web browser plays, written with PyAV (FFmpeg). The
quality setting was chosen by measurement: at CRF 26 a faint object 3x above the noise
vanished completely from the clip, while CRF 22 with grain tuning keeps its brightness.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from .config import ClipConfig
from .models import Event, Frame, ObjectClass
from .render import CLASS_COLORS, _bgr

log = logging.getLogger(__name__)

DEFAULT_SKIP = frozenset({ObjectClass.JUNK, ObjectClass.STAR})


class FrameBuffer:
    """Recent frames at the full frame rate, older frames progressively thinned.

    Frames younger than ``recent`` frames are all kept. Older ones survive only if their
    index is a multiple of 2, 4, 8... once their age passes 1, 2, 4... times ``recent``.
    Memory then grows with the *logarithm* of the time covered, so a two-minute satellite
    pass still has frames to show (as a time-lapse) without storing every frame.
    """

    def __init__(self, recent: int, max_age: int):
        self.recent = max(1, recent)
        self.max_age = max(self.recent, max_age)
        self._frames: deque[tuple[int, np.ndarray]] = deque()

    @staticmethod
    def capacity(recent: int, max_age: int) -> int:
        """Approximate number of frames held once the buffer is full."""
        octaves = math.ceil(math.log2(max(max_age / max(recent, 1), 1.0)))
        return recent + recent // 2 * octaves

    def _keeps(self, index: int, now: int) -> bool:
        age = now - index
        if age < self.recent:
            return True
        if age > self.max_age:
            return False
        stride = 2 ** (int(math.log2(age / self.recent)) + 1)
        return index % stride == 0

    def push(self, index: int, image: np.ndarray) -> None:
        self._frames.append((index, image))
        if index % 8 == 0:
            self._frames = deque(e for e in self._frames if self._keeps(e[0], index))

    def window(self, first: int, last: int) -> list[tuple[int, np.ndarray]]:
        return [e for e in self._frames if first <= e[0] <= last]

    def __len__(self) -> int:
        return len(self._frames)


def pick_frames(
    frames: list[tuple[int, np.ndarray]], max_frames: int
) -> tuple[list[tuple[int, np.ndarray]], int]:
    """Evenly spaced frames for a clip, and the stride between them.

    The stride doubles until every frame at that stride is present, so the clip plays at
    an even speed even where the buffer has thinned out its older part, and until the
    clip is no longer than ``max_frames``. (Sources number frames without gaps.)
    """
    first, last = frames[0][0], frames[-1][0]
    have = {i for i, _ in frames}
    stride = 1
    while stride < 4096:
        expected = last // stride - (first - 1) // stride
        present = sum(1 for i in have if i % stride == 0)
        if present >= expected and present <= max_frames:
            break
        stride *= 2
    return [e for e in frames if e[0] % stride == 0], stride


@dataclass
class _Job:
    event: Event
    frames: list[tuple[int, np.ndarray]]
    stride: int
    scale: float  # buffered frames are this fraction of the source resolution
    fps: float
    stem: Path


class ClipRecorder:
    """Buffers frames and writes a clip, trail image and context image for each event."""

    def __init__(
        self,
        run_dir: str | Path,
        config: ClipConfig,
        fps: float,
        skip: frozenset[ObjectClass] = DEFAULT_SKIP,
    ):
        self.run_dir = Path(run_dir)
        self.out_dir = self.run_dir / "clips"
        self.config = config
        self.fps = fps
        self.skip = skip
        recent = round(config.recent_seconds * fps)
        self.buffer = FrameBuffer(recent, round(config.buffer_seconds * fps))
        self._capacity = FrameBuffer.capacity(recent, self.buffer.max_age)
        self._scale: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clips")
        self._pending: list[Future[None]] = []
        self.written = 0

    def feed(self, frame: Frame) -> None:
        image = frame.image
        if not self.config.color and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._scale is None:
            self._scale = self._choose_scale(image)
        if self._scale < 1.0:
            image = cv2.resize(
                image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA
            )
        self.buffer.push(frame.index, image)

    def _choose_scale(self, image: np.ndarray) -> float:
        """Shrink stored frames only if a full buffer would exceed the memory budget."""
        needed = self._capacity * image.nbytes
        budget = self.config.buffer_mb * 1024 * 1024
        scale = min(1.0, math.sqrt(budget / needed)) if needed else 1.0
        log.info(
            "Clip buffer: up to %d frames (%.0f MB) at %.0f%% resolution",
            self._capacity,
            needed * scale * scale / 1e6,
            100 * scale,
        )
        return scale

    def save(self, event: Event) -> Future[None] | None:
        """Queue the media for ``event`` and record the file names in ``event.media``.
        Returns the job writing them (None if no media will be written)."""
        if event.label in self.skip or self._scale is None:
            return None
        pre = round(self.config.pre_roll_s * self.fps)
        post = round(self.config.post_roll_s * self.fps)
        window = self.buffer.window(event.track.first_frame - pre, event.track.last_frame + post)
        if not window:
            log.warning("Event %d: its frames are no longer buffered; no clip", event.id)
            return None
        frames, stride = pick_frames(window, self.config.max_frames)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.out_dir / f"{event.id:04d}"
        rel = stem.relative_to(self.run_dir).as_posix()
        event.media = {
            "clip": f"{rel}.mp4",
            "thumbnail": f"{rel}.jpg",
            "context": f"{rel}_context.jpg",
            "speedup": stride,
            "partial": window[0][0] > event.track.first_frame,
        }
        job = _Job(event, frames, stride, self._scale, self.fps, stem)
        future = self._executor.submit(_write_media, job, self.config)
        future.add_done_callback(self._report)
        self._pending.append(future)
        return future

    def _report(self, future: Future[None]) -> None:
        if future.exception() is not None:
            log.error("Writing a clip failed: %s", future.exception())
        else:
            self.written += 1

    def close(self) -> None:
        """Wait for clips still being written."""
        self._executor.shutdown(wait=True)


# --------------------------------------------------------------------------- rendering
def _crop_box(
    xs: np.ndarray, ys: np.ndarray, size: float, shape: tuple[int, ...], cfg: ClipConfig, s: float
) -> tuple[int, int, int, int]:
    """A 16:9 region around the whole path, clamped to the image: (x0, y0, x1, y1)."""
    h, w = shape[:2]
    extent = max(xs.max() - xs.min(), ys.max() - ys.min())
    margin = max(cfg.margin_px * s, 0.25 * extent, size)
    cw = max(xs.max() - xs.min() + 2 * margin, cfg.min_crop_px * s)
    ch = max(ys.max() - ys.min() + 2 * margin, cfg.min_crop_px * s * 9 / 16)
    cw, ch = max(cw, ch * 16 / 9), max(ch, cw * 9 / 16)
    cw, ch = min(cw, w), min(ch, h)
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    return x0, y0, x0 + int(cw), y0 + int(ch)


def _stretch(img: np.ndarray, lo: float, gain: float) -> np.ndarray:
    """Brighten the dark night sky for viewing: sky level -> 16, objects scaled up."""
    out = np.clip(16.0 + (img.astype(np.float32) - lo) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out


def _levels(background: np.ndarray, stack: np.ndarray) -> tuple[float, float]:
    lo = float(np.median(background))
    hi = float(np.percentile(stack, 99.9))
    return lo, min(6.0, 220.0 / max(hi - lo, 1.0))


def _brackets(img: np.ndarray, x: float, y: float, half: float, color: tuple[int, ...]) -> None:
    """Four corner brackets around (x, y): marks the object without hiding it."""
    arm = max(4, int(half * 0.6))
    x0, y0, x1, y1 = int(x - half), int(y - half), int(x + half), int(y + half)
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, 2, cv2.LINE_AA)


def _text(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, ...]) -> None:
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), font, size, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, size, color, 1, cv2.LINE_AA)


def _title(event: Event) -> str:
    name = event.identification.name.strip() if event.identification else str(event.label)
    return f"#{event.id} {name} {event.confidence:.0%}"


def _write_media(job: _Job, cfg: ClipConfig) -> None:
    ev, s = job.event, job.scale
    dets = ev.track.detections
    frames_idx = np.array([d.frame_index for d in dets], dtype=float)
    xs = np.array([d.x for d in dets]) * s
    ys = np.array([d.y for d in dets]) * s
    size = max(max(d.bbox[2], d.bbox[3]) for d in dets) * s
    first_img = job.frames[0][1]
    x0, y0, x1, y1 = _crop_box(xs, ys, size, first_img.shape, cfg, s)
    out_w = cfg.width
    out_h = round(out_w * 9 / 16 / 2) * 2  # every card the same shape
    zoom = out_w / max(x1 - x0, 1)
    interp = cv2.INTER_LINEAR if zoom >= 1 else cv2.INTER_AREA
    color = _bgr(CLASS_COLORS[ev.label])

    during = [img for i, img in job.frames if ev.track.first_frame <= i <= ev.track.last_frame]
    during = during or [img for _, img in job.frames]
    background = job.frames[0][1]
    # Layer the event's frames, but only in a corridor along its own path: other objects
    # that crossed the region meanwhile would otherwise appear in its picture too.
    corridor = np.zeros(background.shape[:2], np.uint8)
    path = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(corridor, [path], False, 255, int(max(9, 2 * size + 6)))
    stack = np.maximum.reduce(during)
    inside = corridor > 0 if background.ndim == 2 else (corridor > 0)[..., None]
    trail_full = np.where(inside, stack, background)
    lo, gain = _levels(background[y0:y1, x0:x1], trail_full[y0:y1, x0:x1])

    # Trail image (thumbnail): the whole event layered into one picture.
    crop = trail_full[y0:y1, x0:x1]
    trail = cv2.resize(_stretch(crop, lo, gain), (out_w, out_h), interpolation=interp)
    cv2.imwrite(str(job.stem) + ".jpg", trail, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Context image: the whole view, with the path and the clip's region marked.
    ctx_scale = min(1.0, cfg.width / background.shape[1])
    lo_full, gain_full = _levels(background, trail_full)
    context = cv2.resize(
        _stretch(background, lo_full, gain_full),
        None,
        fx=ctx_scale,
        fy=ctx_scale,
        interpolation=cv2.INTER_AREA,
    )
    cv2.polylines(context, [(path * ctx_scale).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    corner0 = (int(x0 * ctx_scale), int(y0 * ctx_scale))
    corner1 = (int(x1 * ctx_scale), int(y1 * ctx_scale))
    cv2.rectangle(context, corner0, corner1, (200, 200, 200), 1)
    _text(context, _title(ev), (8, 20), color)
    cv2.imwrite(str(job.stem) + "_context.jpg", context, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # The clip: zoomed region, brackets on the object, its trail behind it.
    px = (xs - x0) * zoom
    py = (ys - y0) * zoom
    half = max(10.0, size * zoom / 2 + 6)
    trail_color = tuple(int(c * 0.55) for c in color)
    start = ev.track.start_time
    out_frames = []
    for index, img in job.frames:
        view = _stretch(img[y0:y1, x0:x1], lo, gain)
        view = cv2.resize(view, (out_w, out_h), interpolation=interp)
        done = frames_idx <= index
        if done.sum() >= 2:
            pts = np.stack([px[done], py[done]], axis=1).astype(np.int32)
            cv2.polylines(view, [pts], False, trail_color, 1, cv2.LINE_AA)
        if frames_idx[0] <= index <= frames_idx[-1]:
            cx, cy = np.interp(index, frames_idx, px), np.interp(index, frames_idx, py)
            _brackets(view, cx, cy, half, color)
        when = start + timedelta(seconds=(index - frames_idx[0]) / job.fps)
        _text(view, _title(ev), (8, 20), color)
        stamp = f"{when:%H:%M:%S} UTC" + (f"  x{job.stride}" if job.stride > 1 else "")
        _text(view, stamp, (8, out_h - 10), (220, 220, 220))
        out_frames.append(view)
    write_mp4(Path(str(job.stem) + ".mp4"), out_frames, job.fps, cfg.crf)


def write_mp4(path: Path, frames: list[np.ndarray], fps: float, crf: int = 22) -> None:
    """Write BGR frames as browser-playable H.264 (falls back to OpenCV's MPEG-4)."""
    h, w = frames[0].shape[:2]
    if "libx264" in av.codecs_available:
        with av.open(str(path), "w", options={"movflags": "faststart"}) as container:
            stream: Any = container.add_stream(
                "libx264", rate=Fraction(fps).limit_denominator(1001)
            )
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.options = {"crf": str(crf), "preset": "veryfast", "tune": "grain"}
            for img in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return
    log.warning("FFmpeg lacks an H.264 encoder; writing MPEG-4 (may not play in browsers)")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    for img in frames:
        writer.write(img)
    writer.release()
