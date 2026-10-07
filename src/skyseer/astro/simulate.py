"""Physically correct test skies: real stars and real satellite orbits, known truth.

Used to validate calibration and satellite identification end to end before any camera
hardware exists: the rendered video goes through exactly the same pipeline as real footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from ..synth import _stamp
from .camera import CameraModel
from .catalog import StarCatalog
from .frames import Observer, altaz_to_vec, apply_refraction, icrs_to_enu_matrix
from .orbits import OrbitCatalog, is_sunlit


def camera_looking_at(
    alt: float,
    az: float,
    roll: float,
    width: int,
    height: int,
    hfov: float,
    k1: float = 0.0,
    k2: float = 0.0,
) -> CameraModel:
    """A camera model (ENU frame) pointing at (alt, az) with the image rotated by ``roll``."""
    b = altaz_to_vec(alt, az)
    zen = np.array([0.0, 0.0, 1.0])
    up = zen - (zen @ b) * b
    if np.linalg.norm(up) < 1e-9:
        up = np.array([0.0, 1.0, 0.0]) - b[1] * b
    up /= np.linalg.norm(up)
    right = np.cross(b, up)  # image +x points right when looking along b with "up" up
    down = -up
    r = np.radians(roll)  # camera turned clockwise (seen from behind) by ``roll``
    x_axis = np.cos(r) * right + np.sin(r) * down
    y_axis = -np.sin(r) * right + np.cos(r) * down
    rotation = np.stack([x_axis, y_axis, b], axis=1)  # columns: camera axes in ENU
    f = (width / 2) / np.tan(np.radians(hfov) / 2)
    return CameraModel(width, height, f, width / 2, height / 2, rotation, k1, k2)


def render_starfield(
    camera: CameraModel,
    observer: Observer,
    when: datetime,
    catalog: StarCatalog,
    limit_mag: float = 6.0,
    saturation_mag: float = 2.0,
    sigma: float = 1.3,
    sky: float = 20.0,
    noise: float = 2.5,
    seed: int = 0,
) -> np.ndarray:
    """Grayscale uint8 image of the sky as the given camera would see it."""
    rng = np.random.default_rng(seed)
    img = np.full((camera.height, camera.width), sky, dtype=np.float32)
    cat = catalog.brighter_than(limit_mag)
    v = apply_refraction(cat.vectors() @ icrs_to_enu_matrix(when, observer).T)
    uv, front = camera.vec_to_pixel(v)
    vis = camera.in_image(uv, front, margin=-5) & (v[:, 2] > 0)
    amp = 235.0 * 10 ** (-0.4 * (cat.mag[vis] - saturation_mag))
    for (x, y), a in zip(uv[vis], amp, strict=True):
        _stamp(img, float(x), float(y), float(a), sigma)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class SimulatedPass:
    norad_id: int
    name: str
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)


def find_passes(
    orbits: OrbitCatalog,
    observer: Observer,
    camera: CameraModel,
    start: datetime,
    seconds: float,
    min_seconds_in_view: float = 5.0,
    step_s: float = 1.0,
    min_revs_per_day: float = 11.0,
) -> list[tuple[int, float]]:
    """Sunlit satellites that cross the camera's view during the window.

    ``min_revs_per_day`` keeps low-Earth-orbit objects (the ones bright and fast enough for
    a camera to see moving); geostationary and navigation satellites are excluded.
    Returns (catalogue index, seconds in view) sorted by time in view.
    """
    revs = np.array([s.no_kozai * 1440 / (2 * np.pi) for s in orbits.satrecs])
    leo = np.nonzero(revs >= min_revs_per_day)[0]
    orbits_all, orbits = orbits, orbits.subset(leo)
    times = [start + timedelta(seconds=k * step_s) for k in range(int(seconds / step_s) + 1)]
    in_view = np.zeros(len(orbits), dtype=int)
    for chunk in range(0, len(times), 30):
        ts = times[chunk : chunk + 30]
        enu, _, r_teme, valid = orbits.topocentric(ts, observer)
        n, t = valid.shape
        uv, ok = camera.vec_to_pixel(apply_refraction(enu.reshape(-1, 3)))
        vis = camera.in_image(uv, ok, margin=10).reshape(n, t) & valid & (enu[..., 2] > 0)
        for j, when in enumerate(ts):
            vis[:, j] &= is_sunlit(r_teme[:, j], when)
        in_view += vis.sum(axis=1)
    secs = in_view * step_s
    idx = np.nonzero(secs >= min_seconds_in_view)[0]
    del orbits_all  # indices below refer back to the full catalogue
    return sorted(((int(leo[i]), float(secs[i])) for i in idx), key=lambda p: -p[1])


def render_sky_video(
    out_path: str | Path,
    camera: CameraModel,
    observer: Observer,
    start: datetime,
    seconds: float,
    fps: float,
    stars: StarCatalog,
    orbits: OrbitCatalog,
    satellites: list[int],
    satellite_amp: float = 110.0,
    noise: float = 2.5,
    seed: int = 0,
) -> list[SimulatedPass]:
    """Render a fixed-camera video of the real sky with the chosen catalogue satellites.

    Satellites are drawn where SGP4 puts them (apparent, refracted directions) at each
    frame's time. The star layer is re-rendered every ``star_step_s`` seconds so stars drift
    with the Earth's rotation (15 arcsec/s) like in a real fixed camera.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    star_step_s = 2.0
    star_layers: dict[int, np.ndarray] = {}

    def sky_at(t_rel: float) -> np.ndarray:
        block = int(t_rel // star_step_s)
        if block not in star_layers:
            when = start + timedelta(seconds=(block + 0.5) * star_step_s)
            star_layers[block] = render_starfield(camera, observer, when, stars, noise=0.0).astype(
                np.float32
            )
        return star_layers[block]

    n = int(seconds * fps)
    times = [start + timedelta(seconds=k / fps) for k in range(n)]
    sub = orbits.subset(np.array(satellites))
    passes = [SimulatedPass(int(sub.norad[i]), sub.names[i]) for i in range(len(sub))]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (camera.width, camera.height)
    )
    for chunk in range(0, n, 50):
        ts = times[chunk : chunk + 50]
        enu, _, r_teme, valid = sub.topocentric(ts, observer)
        for j, when in enumerate(ts):
            frame_index = chunk + j
            sky = sky_at(frame_index / fps)
            img = sky + rng.normal(0, noise, sky.shape).astype(np.float32)
            uv, ok = camera.vec_to_pixel(apply_refraction(enu[:, j]))
            vis = camera.in_image(uv, ok, margin=3) & valid[:, j] & is_sunlit(r_teme[:, j], when)
            for i in np.nonzero(vis)[0]:
                _stamp(img, float(uv[i, 0]), float(uv[i, 1]), satellite_amp, 1.3)
                passes[i].points.append((frame_index, float(uv[i, 0]), float(uv[i, 1])))
            gray = np.clip(img, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return passes
