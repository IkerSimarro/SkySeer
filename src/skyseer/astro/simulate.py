"""Render physically placed star fields for testing calibration against known truth."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..synth import _stamp
from .camera import CameraModel
from .catalog import StarCatalog
from .frames import Observer, altaz_to_vec, apply_refraction, icrs_to_enu_matrix


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
