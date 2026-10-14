"""Synthetic night-sky videos with exact ground truth.

Real labelled sky footage is scarce, so tests, benchmarks and the classifier's training
set come from here: a static star field with twinkle, sky glow and sensor noise, plus
satellites, meteors and planes with known trajectories.

The default scenes are deliberately clean (the standard benchmark). ``realistic_spec``
draws *difficult* scenes instead, a technique called domain randomisation: faint and
partial objects, tumbling and flaring satellites, varied aircraft strobes, and the
things that fool a detector on real footage (insects near the lens, twinkling stars,
hot pixels, drifting moonlit clouds, video compression, camera wobble), at several
resolutions and frame rates. A classifier trained across that variety has to learn what
really separates the classes rather than the quirks of one simulator setting.
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
    shake_px: float = 0.0  # camera wobble amplitude (wind on a mast), 0 = rock steady
    shake_hz: float = 2.0  # typical wobble frequency
    # Realism (all off by default, so the standard benchmark scenes never change):
    variety: bool = False  # varied durations, speeds, tumbling, flares, strobe patterns
    brightness: float = 1.0  # scales object brightness (< 1: fainter objects)
    n_insects: int = 0  # curved, blurry streaks close to the lens (truth label Junk)
    n_twinklers: int = 0  # strongly scintillating stars (Junk)
    n_hot_pixels: int = 0  # flickering sensor pixels (Junk)
    n_clouds: int = 0  # drifting moonlit cloud patches (Junk)
    jpeg_quality: int = 0  # > 0: compress every frame like streamed/re-encoded video
    lens: float = 1.0  # on-screen speed factor of sky objects: < 1 = wider field of view
    repeat: int = 1  # each picture is shown this many frames (low native frame rate, time-lapse)


@dataclass
class TruthObject:
    label: str
    first_frame: int
    last_frame: int
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)
    kind: str = ""  # what a Junk object is (insect, twinkler, hot_pixel, cloud)


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


class _VariedMover(_Mover):
    """A satellite, meteor or plane with the variety of real ones (``SceneSpec.variety``).

    Satellites may tumble (slow, deep brightness swings: a rocket body) or flare (a
    brief glint off a panel); meteors vary in width and may burst; aircraft strobes vary
    in rate, may double-flash, and the steady light may be too faint to see. Paths may
    cross the edge of the view, so the truth only lists the frames where it is visible.
    """

    def __init__(
        self,
        label: ObjectClass,
        start: int,
        frames: int,
        p0: np.ndarray,
        v: np.ndarray,
        amp: float,
        rng: np.random.Generator,
        fps: float,
    ):
        super().__init__(label, start, frames, p0, v, amp, rng)
        self.sigma = 1.3
        self.tumble: tuple[float, float, float] | None = None  # period (frames), depth, phase
        self.flare: tuple[float, float, float] | None = None  # frame, width (frames), gain
        if label is ObjectClass.SATELLITE:
            if rng.random() < 0.25:
                period = rng.uniform(2, 10) * fps
                self.tumble = (period, rng.uniform(0.3, 0.9), rng.uniform(0, 2 * math.pi))
            if rng.random() < 0.15:
                at = start + rng.uniform(0.2, 0.8) * frames
                self.flare = (at, rng.uniform(0.1, 0.4) * fps, rng.uniform(1.5, 5.0))
        elif label is ObjectClass.METEOR:
            self.sigma = rng.uniform(1.0, 2.0)
            if rng.random() < 0.3:
                at = start + rng.uniform(0.4, 0.9) * frames
                self.flare = (at, max(1.0, 0.05 * fps), rng.uniform(0.5, 2.0))
        else:
            self.strobe_hz = rng.uniform(0.6, 1.4)
            self.flash_frames = max(1, round(0.08 * fps))
            self.double = rng.random() < 0.4
            steady = rng.uniform(-0.2, 0.8)
            self.steady = steady if steady > 0.1 else 0.0  # sometimes only strobes show
            self.strobe_gain = rng.uniform(1.0, 2.5)
            self.offset = rng.uniform(0, 4)
            self.phase = rng.uniform(0, 1)

    def _gain(self, f: int) -> float:
        g = 1.0
        if self.tumble is not None:
            period, depth, phase = self.tumble
            g *= 1 - depth * (0.5 + 0.5 * math.sin(2 * math.pi * (f - self.start) / period + phase))
        if self.flare is not None:
            at, width, gain = self.flare
            g += gain * math.exp(-0.5 * ((f - at) / width) ** 2)
        return g

    def draw(self, img: np.ndarray, f: int, fps: float) -> None:
        if not (self.start <= f < self.start + self.frames):
            return
        p = self.pos(f)
        h, w = img.shape
        # A meteor is a streak each frame; its centre (brighter towards the head) is what
        # the detector measures, up to ~20 px behind the head for the fastest ones.
        c = p - 0.4 * (p - self.pos(f - 1)) if self.label is ObjectClass.METEOR else p
        if 0 <= c[0] < w and 0 <= c[1] < h:
            self.truth.points.append((f, float(c[0]), float(c[1])))
        amp = self.amp * self._gain(f)
        if self.label is ObjectClass.METEOR:
            phase = (f - self.start) / max(self.frames - 1, 1)
            amp *= math.sin(math.pi * (0.15 + 0.85 * phase))
            tail = self.pos(f - 1)
            steps = max(2, int(np.linalg.norm(p - tail)))
            for t in np.linspace(0, 1, steps):
                q = tail + (p - tail) * t
                _stamp(img, q[0], q[1], amp * (0.4 + 0.6 * t), self.sigma)
        elif self.label is ObjectClass.PLANE:
            if self.steady:
                _stamp(img, p[0], p[1], amp * self.steady, 1.4)
            beat = ((f - self.start) / fps * self.strobe_hz + self.phase) % 1.0
            flash = beat * fps / self.strobe_hz < self.flash_frames
            if self.double:
                second = (beat - 0.15 * self.strobe_hz) % 1.0
                flash = flash or second * fps / self.strobe_hz < self.flash_frames
            if flash:
                _stamp(img, p[0] + self.offset, p[1], amp * self.strobe_gain, 1.6)
        else:
            _stamp(img, p[0], p[1], amp, self.sigma)


class _Insect(_Mover):
    """An insect or bat close to the lens: a fast, blurry, curving, flickering streak."""

    def __init__(
        self,
        start: int,
        frames: int,
        p0: np.ndarray,
        rng: np.random.Generator,
        fps: float,
        scale: float,
        brightness: float,
    ):
        # Often only gently curved: those are the ones that resemble meteors.
        speed = rng.uniform(3, 25) * scale
        heading = rng.uniform(0, 2 * math.pi)
        turn = rng.choice([-1, 1]) * math.radians(rng.uniform(5, 120)) / fps
        wiggle, wiggle_hz = rng.uniform(0, 6) * scale, rng.uniform(1, 6)
        k = np.arange(frames + 1)
        angles = heading + turn * k
        steps = speed * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        normal = np.stack([-np.sin(angles), np.cos(angles)], axis=1)
        sway = wiggle * np.sin(2 * math.pi * wiggle_hz * k / fps)
        self.path = p0 + np.cumsum(steps, axis=0) + normal * sway[:, None]
        self.flicker = rng.uniform(0.6, 1.4, frames + 1)
        amp = rng.uniform(20, 150) * brightness
        super().__init__(ObjectClass.JUNK, start, frames, p0, np.zeros(2), amp, rng)
        self.sigma = rng.uniform(1.2, 3.0) * scale
        self.truth.kind = "insect"

    def pos(self, f: int) -> np.ndarray:
        return self.path[min(max(f - self.start, 0), len(self.path) - 1)]

    def draw(self, img: np.ndarray, f: int, fps: float) -> None:
        if not (self.start <= f < self.start + self.frames):
            return
        p, tail = self.pos(f), self.pos(f - 1)
        c = (p + tail) / 2  # centre of this frame's streak
        h, w = img.shape
        if 0 <= c[0] < w and 0 <= c[1] < h:
            self.truth.points.append((f, float(c[0]), float(c[1])))
        amp = self.amp * self.flicker[f - self.start]
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(p - tail)))):
            q = tail + (p - tail) * t
            _stamp(img, q[0], q[1], amp, self.sigma)


class _Flicker(_Mover):
    """A fixed point that flickers: a scintillating star or a hot sensor pixel."""

    def __init__(
        self, kind: str, p0: np.ndarray, frames: int, rng: np.random.Generator, fps: float
    ):
        super().__init__(ObjectClass.JUNK, 0, frames, p0, np.zeros(2), 0.0, rng)
        self.truth.kind = kind
        if kind == "twinkler":
            # Strong scintillation near the horizon: brightness wanders by tens of percent
            # within a fraction of a second (a smooth random process, AR(1)).
            self.amp, self.sigma = rng.uniform(40, 150), 1.1
            phi = math.exp(-1 / (rng.uniform(0.05, 0.3) * fps))
            walk = np.zeros(frames)
            for i in range(1, frames):
                walk[i] = phi * walk[i - 1] + math.sqrt(1 - phi * phi) * rng.normal()
            self.level = np.clip(1 + 0.7 * walk, 0.05, 2.5)
        else:  # hot pixel: randomly on or off
            self.amp, self.sigma = rng.uniform(40, 200), 0.5
            self.level = (rng.random(frames) < rng.uniform(0.05, 0.3)).astype(float)

    def draw(self, img: np.ndarray, f: int, fps: float) -> None:
        if f < len(self.level) and self.level[f] > 0:
            self.truth.points.append((f, float(self.p0[0]), float(self.p0[1])))
            _stamp(img, self.p0[0], self.p0[1], self.amp * self.level[f], self.sigma)


class _Cloud:
    """A drifting patch of moonlit cloud: faint smooth texture that dims stars behind it."""

    def __init__(self, spec: SceneSpec, rng: np.random.Generator):
        scale = spec.width / 960
        size = int(rng.uniform(80, 260) * scale)
        noise = rng.normal(0, 1, (size, size)).astype(np.float32)
        tex = cv2.GaussianBlur(noise, (0, 0), size / 8)
        tex = (tex - tex.min()) / max(float(tex.max() - tex.min()), 1e-6)
        yy, xx = np.mgrid[0:size, 0:size] / (size - 1) - 0.5
        self.opacity = (tex * np.exp(-(xx**2 + yy**2) / 0.08)).astype(np.float32)
        self.amp = rng.uniform(4, 18)
        ang = rng.uniform(0, 2 * math.pi)
        self.v = rng.uniform(0.2, 1.2) * scale * np.array([math.cos(ang), math.sin(ang)])
        centre = rng.uniform([0, 0], [spec.width, spec.height])
        self.p0 = centre - self.v * spec.seconds * spec.fps / 2

    def add(self, layer: np.ndarray, f: int) -> None:
        """Add this cloud's opacity (0..1) at frame ``f`` into ``layer``, in place."""
        size = self.opacity.shape[0]
        x, y = self.p0 + self.v * f - size / 2
        m = np.array([[1, 0, x - math.floor(x)], [0, 1, y - math.floor(y)]], np.float32)
        patch = cv2.warpAffine(self.opacity, m, (size, size))
        x0, y0 = math.floor(x), math.floor(y)
        h, w = layer.shape
        ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + size, w), min(y0 + size, h)
        if ax0 < ax1 and ay0 < ay1:
            layer[ay0:ay1, ax0:ax1] += patch[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]


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


def _varied_scene(spec: SceneSpec) -> list[_Mover]:
    """Objects with realistic variety: durations, speeds and brightness spread widely,
    and satellite and aircraft paths may cross the edge of the view."""
    rng = np.random.default_rng(spec.seed)
    fps, total = spec.fps, int(spec.seconds * spec.fps)
    earliest = int(spec.warmup_seconds * fps)
    scale = spec.width / 960 * 30 / fps  # px per frame scales with resolution and rate
    movers: list[_Mover] = []

    def add(
        label: ObjectClass,
        seconds: tuple[float, float],
        speed: tuple[float, float],
        amp: tuple[float, float],
        crossing: float,
    ) -> None:
        frames = min(int(rng.uniform(*seconds) * fps), total - earliest - 1)
        start = int(rng.integers(earliest, max(earliest + 1, total - frames)))
        s = rng.uniform(*speed) * scale * spec.lens  # a wider lens shows slower motion
        try:
            if rng.random() < crossing:
                raise RuntimeError
            p0, v = _place(rng, spec, s, frames)
        except RuntimeError:  # crosses the edge of the view
            ang = rng.uniform(0, 2 * math.pi)
            v = s * np.array([math.cos(ang), math.sin(ang)])
            p0 = rng.uniform([0, 0], [spec.width, spec.height]) - v * rng.uniform(0, frames)
        a = rng.uniform(*amp) * spec.brightness
        movers.append(_VariedMover(label, start, frames, p0, v, a, rng, fps))

    for _ in range(spec.n_satellites):
        add(ObjectClass.SATELLITE, (2, 25), (0.3, 3.0), (20, 160), 0.4)
    for _ in range(spec.n_meteors):
        add(ObjectClass.METEOR, (0.15, 2.0), (4, 40), (30, 230), 0.1)
    for _ in range(spec.n_planes):
        add(ObjectClass.PLANE, (6, 25), (0.4, 2.5), (40, 140), 0.4)
    return movers


def _distractors(spec: SceneSpec) -> list[_Mover]:
    """Things that move or flicker but aren't satellites, meteors or aircraft."""
    rng = np.random.default_rng(spec.seed + 4)
    fps, total = spec.fps, int(spec.seconds * spec.fps)
    scale = spec.width / 960 * 30 / fps
    size = np.array([spec.width, spec.height])
    out: list[_Mover] = []
    for _ in range(spec.n_insects):
        frames = int(rng.uniform(0.2, 1.5) * fps)
        start = int(rng.integers(int(spec.warmup_seconds * fps), max(total - frames, 1)))
        p0 = rng.uniform([0, 0], size)
        out.append(_Insect(start, frames, p0, rng, fps, scale, spec.brightness))
    for kind, n in (("twinkler", spec.n_twinklers), ("hot_pixel", spec.n_hot_pixels)):
        for _ in range(n):
            p0 = np.round(rng.uniform([10, 10], size - 10))
            out.append(_Flicker(kind, p0, total, rng, fps))
    return out


def build_scene(spec: SceneSpec) -> list[_Mover]:
    if spec.variety:
        return _varied_scene(spec)
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


def _shake_path(spec: SceneSpec) -> np.ndarray | None:
    """Per-frame (dx, dy) camera wobble: two swaying motions per axis, like wind on a mast."""
    if spec.shake_px <= 0:
        return None
    rng = np.random.default_rng(spec.seed + 2)
    t = np.arange(int(spec.seconds * spec.fps)) / spec.fps
    out = np.zeros((t.size, 2))
    for axis in range(2):
        for weight in (0.7, 0.3):
            freq = spec.shake_hz * rng.uniform(0.6, 1.6)
            out[:, axis] += weight * np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))
    return spec.shake_px * out


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

    movers = build_scene(spec) + _distractors(spec)
    cloud_rng = np.random.default_rng(spec.seed + 5)
    clouds = [_Cloud(spec, cloud_rng) for _ in range(spec.n_clouds)]
    shake = _shake_path(spec)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), spec.fps, (w, h))
    held: np.ndarray | None = None
    for f in range(int(spec.seconds * spec.fps)):
        if spec.repeat > 1 and f % spec.repeat and held is not None:
            writer.write(held)  # the same picture again (video made at a lower frame rate)
            continue
        twinkle = 1.0 + rng.normal(0, 0.04)
        noise = rng.normal(0, spec.noise_sigma, (h, w)).astype(np.float32)
        if clouds:  # cloud dims the stars behind it and glows faintly in the moonlight
            cover = np.zeros((h, w), np.float32)
            for c in clouds:
                c.add(cover, f)
            cover = np.minimum(cover, 1.0)
            glow = sum(c.amp for c in clouds) / len(clouds) * cover
            img = sky + glow + stars * twinkle * (1 - 0.7 * cover)
        else:
            img = sky + stars * twinkle
        if shake is None:
            img = img + noise
            for m in movers:
                m.draw(img, f, spec.fps)
        else:  # the whole scene wobbles; the sensor noise does not
            for m in movers:
                m.draw(img, f, spec.fps)
            shift = np.array([[1, 0, shake[f, 0]], [0, 1, shake[f, 1]]], np.float32)
            img = cv2.warpAffine(img, shift, (w, h), borderMode=cv2.BORDER_REFLECT) + noise
        gray = np.clip(img, 0, 255).astype(np.uint8)
        if spec.jpeg_quality:
            _, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality])
            gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        held = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        writer.write(held)
    writer.release()
    if spec.repeat > 1:  # objects stay visible, unmoved, while a picture is held
        for m in movers:
            m.truth.points = [
                (f + k, x, y) for f, x, y in m.truth.points for k in range(spec.repeat)
            ]

    objects = []
    for m in movers:  # the truth covers only the frames where each object was in view
        if m.truth.points:
            m.truth.first_frame, m.truth.last_frame = m.truth.points[0][0], m.truth.points[-1][0]
            objects.append(m.truth)
    truth = GroundTruth(spec, objects)
    truth.save(out_path.with_suffix(".json"))
    return truth


def realistic_spec(seed: int, seconds: float = 20.0) -> SceneSpec:
    """A randomly drawn *difficult* night (domain randomisation, see the module notes)."""
    rng = np.random.default_rng(10_000 + seed)
    w, h = [(640, 360), (960, 540), (1280, 720)][int(rng.choice(3, p=[0.4, 0.4, 0.2]))]
    area = w * h / (960 * 540)
    return SceneSpec(
        width=w,
        height=h,
        fps=float(rng.choice([25.0, 30.0])),
        seconds=seconds,
        seed=seed,
        n_stars=int(250 * area * rng.uniform(0.3, 2.0)),
        n_satellites=int(rng.integers(0, 5)),
        n_meteors=int(rng.integers(0, 4)),
        n_planes=int(rng.integers(0, 3)),
        noise_sigma=float(rng.uniform(1, 6)),
        shake_px=float(rng.uniform(0.3, 2.5)) if rng.random() < 0.3 else 0.0,
        variety=True,
        brightness=float(rng.uniform(0.3, 1.0)),
        n_insects=int(rng.integers(0, 4)),
        n_twinklers=int(rng.integers(0, 4)),
        n_hot_pixels=int(rng.integers(0, 5)),
        n_clouds=int(rng.integers(1, 3)) if rng.random() < 0.3 else 0,
        jpeg_quality=int(rng.integers(35, 86)) if rng.random() < 0.5 else 0,
        # Lenses from telephoto-ish to very wide: wide views show everything moving slowly.
        lens=float(np.exp(rng.uniform(np.log(0.15), np.log(1.2)))),
        repeat=int(rng.choice([2, 3, 5])) if rng.random() < 0.25 else 1,
    )
