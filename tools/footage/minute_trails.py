"""Per-minute trail images: everything that moved in each minute, regardless of what
SkySeer detected. Compare with SkySeer's events to spot what it missed.

    python tools/footage/minute_trails.py VIDEO OUT.jpg --from-min 0 --to-min 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--from-min", type=float, default=0.0)
    ap.add_argument("--to-min", type=float, default=12.0)
    ap.add_argument("--gain", type=float, default=6.0, help="brightness boost")
    args = ap.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.from_min * 60 * fps))
    tiles, minute = [], args.from_min
    while minute < args.to_min:
        frames = []
        for i in range(int(60 * fps)):
            if not cap.grab():
                break
            if i % 2 == 0:
                frames.append(cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY))
        if not frames:
            break
        st = np.stack(frames)
        trail = st.max(0).astype(np.float32) - np.median(st[::10], 0)
        img = cv2.cvtColor(np.clip(trail * args.gain, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(img, f"{minute:.0f}-{minute + 1:.0f} min", (6, 20), 0, 0.6, (0, 255, 255), 2)
        tiles.append(img)
        minute += 1
    cap.release()
    cols = 4
    rows = [
        np.hstack(
            tiles[i : i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i : i + cols]))
        )
        for i in range(0, len(tiles), cols)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"{len(tiles)} minutes -> {args.out}")


if __name__ == "__main__":
    main()
