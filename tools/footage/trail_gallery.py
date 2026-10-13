"""Gallery of 'trail images': every frame of each event layered into one picture.

    python tools/footage/trail_gallery.py VIDEO EVENTS.json OUT.jpg [--labels Meteor,Plane]
                                          [--min-duration 3] [--max 36] [--every 1]

A satellite appears as a smooth line, an aircraft as a dotted (strobing) line, a meteor
as a short bright streak, and junk as blobs, wiggles or cloud texture, which makes
it quick to check classifications by eye. Each tile lists class, confidence, start time,
duration, speed, straightness and strobe score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = {
    "Satellite": (77, 77, 255),
    "Meteor": (63, 210, 255),
    "Plane": (255, 171, 77),
    "Star": (177, 165, 154),
    "Junk": (128, 128, 128),
}


def tile(cap: cv2.VideoCapture, e: dict, fps: float, w: int, h: int) -> np.ndarray | None:
    f0, f1 = e["first_frame"], e["last_frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0 - 3))
    step = max(1, (f1 - f0 + 6) // 300)
    stack = None
    for i in range(f1 - f0 + 7):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stack = g if stack is None else np.maximum(stack, g)
    if stack is None:
        return None
    xs = [p["x"] for p in e["track"]]
    ys = [p["y"] for p in e["track"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(40, int(max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 25))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = stack[y0:y1, x0:x1].astype(np.float32)
    crop = np.clip((crop - np.percentile(crop, 50)) * 4 + 20, 0, 255).astype(np.uint8)
    canvas = np.zeros((310, 240, 3), np.uint8)
    canvas[:240] = cv2.resize(
        cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), (240, 240), interpolation=cv2.INTER_NEAREST
    )
    f = e["features"]
    name = (e.get("identification") or {}).get("name", "")
    shower = e.get("shower")
    extra = name or (shower["code"] if isinstance(shower, dict) else (shower or ""))
    mag = (e.get("photometry") or {}).get("peak_magnitude")
    lines = [
        f"#{e['id']} {e['label']} {e['confidence']:.2f} {extra}"[:34],
        f"t={f0 / fps / 60:.1f}min {e['duration_s']:.1f}s"
        + (f" mag {mag:+.1f}" if mag is not None else ""),
        f"spd {f['avg_speed']:.1f}px/f dev {f.get('path_deviation', 0):.3f}",
        f"strobe {f.get('flash_score', 0):.2f} n={int(f['detection_count'])}",
    ]
    for k, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (4, 256 + k * 15),
            0,
            0.42,
            COLORS.get(e["label"], (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("events", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", default=None, help="comma-separated classes to include")
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--max", type=int, default=36, help="maximum number of tiles")
    ap.add_argument("--every", type=int, default=1, help="take every Nth matching event")
    args = ap.parse_args()
    events = json.loads(args.events.read_text())
    if args.labels:
        wanted = set(args.labels.split(","))
        events = [e for e in events if e["label"] in wanted]
    events = [e for e in events if e["duration_s"] >= args.min_duration][:: args.every][: args.max]
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    tiles = [t for e in events if (t := tile(cap, e, fps, w, h)) is not None]
    cap.release()
    if not tiles:
        raise SystemExit("No events to show")
    cols = 6
    rows = [
        np.hstack(
            tiles[i : i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i : i + cols]))
        )
        for i in range(0, len(tiles), cols)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"{len(tiles)} events -> {args.out}")


if __name__ == "__main__":
    main()
