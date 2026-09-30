"""Synthetic night-sky videos with exact ground truth.

Real labelled sky footage is scarce, so tests, benchmarks and the first classifier
training set come from here: a static star field with twinkle, sky glow and sensor
noise, plus satellites, meteors and planes with known trajectories.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .models import ObjectClass


@dataclass
class SceneSpec:
    width: int = 960
    height: int = 540
    fps: float = 30.0
    seconds: float = 20.0
    seed: int = 0
    n_stars: int = 250
    n_satellites: int = 2
    n_meteors: int = 2
    n_planes: int = 1
    noise_sigma: float = 3.0
    warmup_seconds: float = 1.5  # objects appear after the background model settles


@dataclass
class TruthObject:
    label: str
    first_frame: int
    last_frame: int
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)


@dataclass
class GroundTruth:
    spec: SceneSpec
    objects: list[TruthObject]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        objects = [
            TruthObject(**{**o, "points": [tuple(p) for p in o["points"]]}) for o in data["objects"]
        ]
        return cls(SceneSpec(**data["spec"]), objects)


def _stamp(img: np.ndarray, x: float, y: float, amp: float, sigma: float) -> None:
    """Add a Gaussian point source to a float image, in place."""
    r = max(2, int(3 * sigma + 1))
    x0, y0 = int(round(x)), int(round(y))
    ys, xs = np.mgrid[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    h, w = img.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not ok.any():
        return
    g = amp * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma * sigma))
    img[ys[ok], xs[ok]] += g[ok]


class _Mover:
    def __init__(
        self,
        label: ObjectClass,
        start: int,
        frames: int,
        p0: np.ndarray,
        v: np.ndarray,
        amp: float,
        rng: np.random.Generator,
    ):
        self.label, self.start, self.frames = label, start, frames
        self.p0, self.v, self.amp = p0, v, amp
        self.blink_phase = rng.integers(0, 30)
        self.truth = TruthObject(str(label), start, start + frames - 1)

    def pos(self, f: int) -> np.ndarray:
        return self.p0 + self.v * (f - self.start)

    def draw(self, img: np.ndarray, f: int, fps: float) -> None:
        if not (self.start <= f < self.start + self.frames):
            return
        p = self.pos(f)
        self.truth.points.append((f, float(p[0]), float(p[1])))
        if self.label is ObjectClass.METEOR:
            # Brightness rises then fades; drawn as a streak over the last frame's motion.
            phase = (f - self.start) / max(self.frames - 1, 1)
            amp = self.amp * math.sin(math.pi * (0.15 + 0.85 * phase))
            tail = self.pos(f - 1)
            steps = max(2, int(np.linalg.norm(p - tail)))
            for t in np.linspace(0, 1, steps):
                q = tail + (p - tail) * t
                _stamp(img, q[0], q[1], amp * (0.4 + 0.6 * t), 1.2)
        elif self.label is ObjectClass.PLANE:
            # Steady navigation light plus a white strobe ~1 Hz.
            _stamp(img, p[0], p[1], self.amp * 0.6, 1.4)
            if (f + self.blink_phase) % int(fps) < 3:
                _stamp(img, p[0] + 3, p[1], self.amp * 1.6, 1.6)
        else:
            _stamp(img, p[0], p[1], self.amp, 1.3)


def _place(
    rng: np.random.Generator, spec: SceneSpec, speed: float, frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random start and velocity keeping the whole path inside the frame (with margin)."""
    m = 20
    for _ in range(1000):
        p0 = rng.uniform([m, m], [spec.width - m, spec.height - m])
        ang = rng.uniform(0, 2 * math.pi)
        v = speed * np.array([math.cos(ang), math.sin(ang)])
        end = p0 + v * frames
        if m <= end[0] <= spec.width - m and m <= end[1] <= spec.height - m:
            return p0, v
    raise RuntimeError("Could not place object path inside the frame; enlarge the scene")


def build_scene(spec: SceneSpec) -> list[_Mover]:
    rng = np.random.default_rng(spec.seed)
    fps, total = spec.fps, int(spec.seconds * spec.fps)
    earliest = int(spec.warmup_seconds * fps)
    movers: list[_Mover] = []

    def add(
        label: ObjectClass,
        seconds: tuple[float, float],
        speed: tuple[float, float],
        amp: tuple[float, float],
    ) -> None:
        frames = min(int(rng.uniform(*seconds) * fps), total - earliest - 1)
        start = int(rng.integers(earliest, max(earliest + 1, total - frames)))
        s = rng.uniform(*speed)
        p0, v = _place(rng, spec, s, frames)
        movers.append(_Mover(label, start, frames, p0, v, rng.uniform(*amp), rng))

    for _ in range(spec.n_satellites):
        add(ObjectClass.SATELLITE, (5, 14), (0.8, 2.5), (70, 160))
    for _ in range(spec.n_meteors):
        add(ObjectClass.METEOR, (0.4, 1.0), (12, 25), (150, 230))
    for _ in range(spec.n_planes):
        add(ObjectClass.PLANE, (10, 16), (0.8, 2.0), (80, 120))
    return movers


def generate(spec: SceneSpec, out_path: str | Path) -> GroundTruth:
    """Render the scene to ``out_path`` (mp4) and save ground truth next to it (.json)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed + 1)
    w, h = spec.width, spec.height

    # Static sky: vertical glow gradient + star field.
    yy = np.linspace(0, 1, h)[:, None]
    sky = np.repeat(18 + 14 * yy, w, axis=1).astype(np.float32)
    stars = np.zeros((h, w), np.float32)
    star_xy = rng.uniform([0, 0], [w, h], size=(spec.n_stars, 2))
    star_amp = rng.pareto(2.5, spec.n_stars) * 25 + 15
    for (x, y), a in zip(star_xy, star_amp, strict=True):
        _stamp(stars, x, y, float(min(a, 230)), 1.0)

    movers = build_scene(spec)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), spec.fps, (w, h))
    for f in range(int(spec.seconds * spec.fps)):
        twinkle = 1.0 + rng.normal(0, 0.04)
        img = sky + stars * twinkle + rng.normal(0, spec.noise_sigma, (h, w)).astype(np.float32)
        for m in movers:
            m.draw(img, f, spec.fps)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()

    truth = GroundTruth(spec, [m.truth for m in movers])
    truth.save(out_path.with_suffix(".json"))
    return truth
