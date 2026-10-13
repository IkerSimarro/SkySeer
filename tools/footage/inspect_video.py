"""First look at a night-sky video: properties, scene changes and an overview image.

    python tools/footage/inspect_video.py samples/night1.mp4 [--out data/runs/night1]

Prints resolution, frame rate, duration and the times where the whole picture changes
(camera moved, cuts or fades in compilation videos), and saves overview.jpg with frames
spread across the video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def scene_changes(path: Path, step: int = 5) -> list[tuple[float, float]]:
    """(time_s, change) where the downscaled picture changes far more than usual."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, diffs, idx = None, [], 0
    while cap.grab():
        if idx % step == 0:
            _, img = cap.retrieve()
            small = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            if prev is not None:
                diffs.append((idx / fps, float(np.abs(small - prev).mean())))
            prev = small
        idx += 1
    cap.release()
    if not diffs:
        return []
    d = np.array([v for _, v in diffs])
    med, mad = np.median(d), np.median(np.abs(d - np.median(d))) + 1e-6
    hits = [(t, v) for t, v in diffs if v > med + 15 * mad and v > 1.0]
    merged: list[tuple[float, float]] = []  # one entry per change (fades span many frames)
    for t, v in hits:
        if merged and t - merged[-1][0] < 3.0:
            if v > merged[-1][1]:
                merged[-1] = (merged[-1][0], v)
        else:
            merged.append((t, v))
    return merged


def overview(path: Path, out: Path, tiles: int = 8) -> None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(0, max(n - 30, 0), tiles).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
            cv2.putText(img, f"{f / fps / 60:.1f} min", (8, 22), 0, 0.6, (0, 255, 255), 2)
            frames.append(img)
    cap.release()
    rows = [
        np.hstack(frames[i : i + 2] + [np.zeros_like(frames[0])] * (2 - len(frames[i : i + 2])))
        for i in range(0, len(frames), 2)
    ]
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overview.jpg"), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path("data/runs") / args.video.stem
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    size_mb = args.video.stat().st_size / 1e6
    print(
        f"{args.video.name}: {w}x{h}, {fps:.2f} fps, {n} frames = {n / fps / 60:.1f} min, "
        f"{size_mb:.0f} MB ({size_mb * 8 / max(n / fps, 1):.1f} Mbit/s)"
    )
    if w < 1280:
        print(
            "  note: low resolution - faint satellites will be tiny and star recognition may fail"
        )
    changes = scene_changes(args.video)
    if changes:
        print(
            f"  {len(changes)} scene change(s)/camera movement(s): "
            + ", ".join(f"{int(t // 60)}:{int(t % 60):02d}" for t, _ in changes)
        )
    else:
        print("  no scene changes: looks like one fixed camera")
    overview(args.video, out)
    print(f"  overview image: {out / 'overview.jpg'}")


if __name__ == "__main__":
    main()
