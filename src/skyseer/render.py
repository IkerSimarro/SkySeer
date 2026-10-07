"""Annotated output video."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .models import Event, ObjectClass

log = logging.getLogger(__name__)

# The single source of truth for class colours (hex, used by the video and the dashboard).
CLASS_COLORS: dict[ObjectClass, str] = {
    ObjectClass.SATELLITE: "#ff4d4d",
    ObjectClass.METEOR: "#ffd23f",
    ObjectClass.PLANE: "#4dabff",
    ObjectClass.STAR: "#9aa5b1",
    ObjectClass.JUNK: "#6b7280",
}


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def _boxes_by_frame(
    events: list[Event], labels: set[ObjectClass] | None
) -> dict[int, list[tuple[Event, tuple[int, int, int, int]]]]:
    """Box for every frame each event spans, linearly interpolated between detections
    (detections only exist on processed frames when frame skipping is on)."""
    out: dict[int, list[tuple[Event, tuple[int, int, int, int]]]] = {}
    for ev in events:
        if labels is not None and ev.label not in labels:
            continue
        dets = ev.track.detections
        frames = np.array([d.frame_index for d in dets])
        boxes = np.array([d.bbox for d in dets], dtype=float)
        for f in range(frames[0], frames[-1] + 1):
            box = tuple(int(np.interp(f, frames, boxes[:, k])) for k in range(4))
            out.setdefault(f, []).append((ev, box))  # type: ignore[arg-type]
    return out


def _dashed_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    dash: float = 8.0,
    gap: float = 6.0,
) -> None:
    """Dashed line through ``pts`` with dashes of constant on-screen length."""
    if len(pts) < 2:
        return
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    pos = np.concatenate([[0.0], np.cumsum(seg_len)])
    period = dash + gap
    for s in np.arange(0.0, pos[-1], period):
        a = np.array([np.interp(s, pos, pts[:, k]) for k in (0, 1)])
        b = np.array([np.interp(min(s + dash, pos[-1]), pos, pts[:, k]) for k in (0, 1)])
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1, cv2.LINE_AA)


def render_annotated_video(
    video_path: str | Path,
    events: list[Event],
    out_path: str | Path,
    speedup: int = 10,
    labels: set[ObjectClass] | None = None,
    pad: int = 6,
    predictions: dict[int, np.ndarray] | None = None,
) -> Path:
    """Write a fast-forwarded video with every event boxed and labelled.

    Idle stretches play ``speedup`` times faster; frames where an event is visible are
    all kept, so brief meteors are never dropped by the fast-forward.

    ``predictions`` maps event id → (K, 2) pixel polyline of the path an identified
    satellite's orbit predicts; it is drawn dashed while the event is visible, so the
    viewer can see the real object riding along its predicted orbit.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    boxes = _boxes_by_frame(events, labels)

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        active = boxes.get(index)
        if active or index % speedup == 0:
            for ev, (x, y, w, h) in active or []:
                color = _bgr(CLASS_COLORS[ev.label])
                if predictions and ev.id in predictions:
                    _dashed_polyline(frame, predictions[ev.id], color)
                cv2.rectangle(frame, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 2)
                name = ev.identification.name.strip() if ev.identification else str(ev.label)
                cv2.putText(
                    frame,
                    f"#{ev.id} {name}",
                    (x - pad, max(12, y - pad - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
        index += 1
    cap.release()
    writer.release()
    log.info("Wrote annotated video %s", out_path)
    return out_path
