"""Physically correct test skies: real stars, real satellite orbits and aircraft, known truth.

Used to validate calibration and satellite identification end to end before any camera
hardware exists: the rendered video goes through exactly the same pipeline as real footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..synth import _stamp
from .camera import CameraModel
from .catalog import StarCatalog
from .frames import Observer, altaz_to_vec, apply_refraction, icrs_to_enu_matrix, radec_to_vec
from .moon import moon_state
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


STAR_SIGMA = 1.3
STAR_SATURATION_MAG = 2.0


def render_moon(
    camera: CameraModel, observer: Observer, when: datetime, sky: float = 20.0
) -> tuple[np.ndarray, float]:
    """What the Moon adds to a camera's picture, as a float layer to add to the sky.

    - the disc, saturated (a night exposure overexposes even a crescent);
    - glare around it, fading with the square of the distance from the disc;
    - a lens-flare ghost: reflections inside the lens put a faint, blurred copy of the
      Moon on the opposite side of the picture's centre, moving the opposite way;
    - moonlight brightening the whole sky, more for a fuller, higher Moon.

    The brightness figures are illustrative, not measured on a real lens. Returns the
    layer and the sky brightening across the whole picture. Zeros if the Moon is down.
    """
    layer = np.zeros((camera.height, camera.width), dtype=np.float32)
    moon = moon_state(when, observer)
    if not moon.up:
        return layer, 0.0
    lit = moon.fraction
    glow = 30.0 * lit * max(float(np.sin(np.radians(moon.alt))), 0.0)
    layer += glow
    uv, front = camera.vec_to_pixel(moon.enu[None, :])
    if not front[0]:
        return layer, glow
    x, y = (float(c) for c in uv[0])
    r_disc = camera.f * np.radians(moon.radius_deg)
    yy, xx = np.mgrid[0 : camera.height, 0 : camera.width].astype(np.float32)
    d = np.hypot(xx - x, yy - y)
    layer += 400.0 * lit * (r_disc / np.maximum(d, r_disc)) ** 2
    layer[d <= r_disc] = 600.0
    ghost = np.hypot(xx - (2 * camera.cx - x), yy - (2 * camera.cy - y))
    layer += 25.0 * lit * np.exp(-0.5 * (ghost / (3 * r_disc)) ** 2)
    return layer, glow


def flux_of_magnitude(mag: float, sigma: float = STAR_SIGMA) -> float:
    """Total rendered flux of a point source of magnitude ``mag`` (same scale as the stars
    drawn by :func:`render_starfield`)."""
    return 2 * np.pi * sigma**2 * 235.0 * 10 ** (-0.4 * (mag - STAR_SATURATION_MAG))


def _rotate(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))


@dataclass
class SimMeteor:
    """A meteor travelling along a great circle (geometric ENU directions)."""

    start_time: datetime
    duration: float
    start: np.ndarray
    end: np.ndarray
    peak_mag: float
    shower: str | None = None
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)

    def position(self, t: float) -> np.ndarray:
        """Direction at ``t`` seconds after start (spherical interpolation)."""
        f = np.clip(t / self.duration, 0.0, 1.0)
        omega = np.arccos(np.clip(self.start @ self.end, -1.0, 1.0))
        if omega < 1e-9:
            return self.start
        return (np.sin((1 - f) * omega) * self.start + np.sin(f * omega) * self.end) / np.sin(omega)

    def magnitude(self, t: float) -> float | None:
        """Brightens then fades (peak about 60% of the way), like a real meteor."""
        f = t / self.duration
        if not 0.0 <= f <= 1.0:
            return None
        shape = max(np.sin(np.pi * f**0.8), 1e-3)
        return self.peak_mag - 2.5 * float(np.log10(shape))


def shower_meteor(
    shower: Any,
    when: datetime,
    observer: Observer,
    camera: CameraModel,
    rng: np.random.Generator,
    peak_mag: float = 2.0,
    duration: tuple[float, float] = (0.4, 1.0),
    radiant_spread_deg: float = 0.0,
    speed_spread: float = 0.0,
) -> SimMeteor | None:
    """A meteor from ``shower`` that appears in the camera's view at ``when``, with the
    angular speed physics predicts for the shower's entry speed at that sky position.

    ``radiant_spread_deg`` and ``speed_spread`` add the natural scatter of real showers
    (individual meteoroids' radiants spread ~1° around the mean; heights and deceleration
    change the apparent speed by ~10-20%).
    """
    from .frames import vec_to_altaz
    from .showers import expected_angular_speed, solar_longitude

    ra, dec = shower.radiant(solar_longitude(when))
    r = radec_to_vec(ra, dec) @ icrs_to_enu_matrix(when, observer).T
    if radiant_spread_deg > 0:
        tilt = np.cross(r, rng.normal(size=3))
        r = _rotate(r, tilt / np.linalg.norm(tilt), abs(rng.normal(0, radiant_spread_deg)))
    if vec_to_altaz(r)[0] < 5:
        return None
    for _ in range(200):
        uv = rng.uniform(
            [0.2 * camera.width, 0.2 * camera.height], [0.8 * camera.width, 0.8 * camera.height]
        )
        p = camera.pixel_to_vec(uv[None])[0]
        dist = float(np.degrees(np.arccos(np.clip(r @ p, -1, 1))))
        alt = float(vec_to_altaz(p)[0])
        if 15 <= dist <= 80 and alt > 15:
            break
    else:
        return None
    axis = np.cross(r, p)
    axis /= np.linalg.norm(axis)
    dur = float(rng.uniform(*duration))
    speed = expected_angular_speed(shower.v_inf, dist, alt)
    if speed_spread > 0:
        speed *= float(np.clip(rng.normal(1.0, speed_spread), 0.5, 1.5))
    length = speed * dur
    return SimMeteor(
        when,
        dur,
        _rotate(r, axis, dist - length / 2),
        _rotate(r, axis, dist + length / 2),
        peak_mag,
        shower.code,
    )


def sporadic_meteor(
    when: datetime, camera: CameraModel, rng: np.random.Generator, peak_mag: float = 2.0
) -> SimMeteor:
    """A meteor with a random direction and a typical speed (not from any shower)."""
    uv = rng.uniform(
        [0.25 * camera.width, 0.25 * camera.height], [0.75 * camera.width, 0.75 * camera.height]
    )
    p = camera.pixel_to_vec(uv[None])[0]
    axis = np.cross(p, rng.normal(size=3))
    axis /= np.linalg.norm(axis)
    dur = float(rng.uniform(0.4, 1.0))
    length = float(rng.uniform(8, 25)) * dur  # 8-25 deg/s
    return SimMeteor(
        when, dur, _rotate(p, axis, -length / 2), _rotate(p, axis, length / 2), peak_mag, None
    )


def render_meteor_video(
    out_path: str | Path,
    camera: CameraModel,
    observer: Observer,
    start: datetime,
    seconds: float,
    fps: float,
    stars: StarCatalog,
    meteors: list[SimMeteor],
    noise: float = 2.5,
    seed: int = 0,
) -> list[SimMeteor]:
    """Render meteors (streaks with known brightness) over a drifting real star field."""
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers: dict[int, np.ndarray] = {}
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (camera.width, camera.height)
    )
    for k in range(int(seconds * fps)):
        block = int(k / fps // 2)
        if block not in layers:
            when = start + timedelta(seconds=(block + 0.5) * 2)
            layers[block] = render_starfield(camera, observer, when, stars, noise=0.0).astype(
                np.float32
            )
        img = layers[block] + rng.normal(0, noise, layers[block].shape).astype(np.float32)
        t_frame = start + timedelta(seconds=k / fps)
        for m in meteors:
            t1 = (t_frame - m.start_time).total_seconds()
            mag = m.magnitude(t1)
            if mag is None:
                continue
            t0 = max(t1 - 1.0 / fps, 0.0)
            pts, ok = camera.vec_to_pixel(
                apply_refraction(np.stack([m.position(t0), m.position(t1)]))
            )
            if not ok.all():
                continue
            a, b = pts
            n = max(2, int(np.hypot(*(b - a)) * 2))
            amp = flux_of_magnitude(mag) / (n * 2 * np.pi * STAR_SIGMA**2)
            for s in np.linspace(0.0, 1.0, n):
                q = a + (b - a) * s
                _stamp(img, float(q[0]), float(q[1]), amp, STAR_SIGMA)
            if camera.in_image(b[None], np.array([True]))[0]:
                m.points.append((k, float(b[0]), float(b[1])))
        writer.write(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    writer.release()
    return meteors


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
    moon: bool = False,
) -> list[SimulatedPass]:
    """Render a fixed-camera video of the real sky with the chosen catalogue satellites.

    Satellites are drawn where SGP4 puts them (apparent, refracted directions) at each
    frame's time. The star layer is re-rendered every ``star_step_s`` seconds so stars drift
    with the Earth's rotation (15 arcsec/s) like in a real fixed camera. With ``moon`` the
    Moon is drawn where it really is then (:func:`render_moon`), and its light adds noise.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    star_step_s = 2.0
    star_layers: dict[int, tuple[np.ndarray, np.ndarray | float]] = {}

    def sky_at(t_rel: float) -> tuple[np.ndarray, np.ndarray | float]:
        """The sky without noise at ``t_rel``, and the noise level (per pixel with a Moon)."""
        block = int(t_rel // star_step_s)
        if block not in star_layers:
            when = start + timedelta(seconds=(block + 0.5) * star_step_s)
            layer = render_starfield(camera, observer, when, stars, noise=0.0).astype(np.float32)
            sigma: np.ndarray | float = noise
            if moon:
                light, _ = render_moon(camera, observer, when)
                layer += light
                # Photon noise grows with the square root of the light: glare and a moonlit
                # sky are noisier, which is what hides faint objects near the Moon.
                sigma = noise * np.sqrt(1 + np.minimum(light, 255.0) / 20.0)
            star_layers.clear()  # only the current block is needed
            star_layers[block] = (layer, sigma)
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
            sky, sigma = sky_at(frame_index / fps)
            img = sky + (rng.standard_normal(sky.shape) * sigma).astype(np.float32)
            uv, ok = camera.vec_to_pixel(apply_refraction(enu[:, j]))
            vis = camera.in_image(uv, ok, margin=3) & valid[:, j] & is_sunlit(r_teme[:, j], when)
            for i in np.nonzero(vis)[0]:
                _stamp(img, float(uv[i, 0]), float(uv[i, 1]), satellite_amp, 1.3)
                passes[i].points.append((frame_index, float(uv[i, 0]), float(uv[i, 1])))
            gray = np.clip(img, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return passes


# --------------------------------------------------------------------------- aircraft
@dataclass
class SimFlight:
    """An aircraft flying a straight line at constant speed, with lights and broadcasts."""

    icao24: str
    callsign: str
    when: datetime  # time at ``xyz``
    xyz: np.ndarray  # Earth-fixed position (m) at ``when``
    velocity: np.ndarray  # Earth-fixed velocity (m/s)
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)

    def position(self, t: np.ndarray) -> np.ndarray:
        """Earth-fixed positions (T, 3) at Unix times ``t``."""
        dt = np.asarray(t, dtype=float) - self.when.timestamp()
        return self.xyz + dt[:, None] * self.velocity

    def broadcasts(
        self, start: datetime, seconds: float, every_s: float, noise_m: float, rng: Any
    ) -> list[Any]:
        """ADS-B positions every ``every_s`` seconds, with GNSS-like position noise."""
        from ..aircraft import AircraftState, ecef_to_geodetic, enu_basis

        t = start.timestamp() + every_s * np.arange(int(seconds / every_s) + 1)
        t = np.round(t)  # broadcasts carry whole-second times, like OpenSky's
        xyz = self.position(t) + rng.normal(0.0, noise_m, (len(t), 3))
        lat, lon, alt = ecef_to_geodetic(xyz)
        out = []
        for k in range(len(t)):
            e, n, u = enu_basis(float(lat[k]), float(lon[k])) @ self.velocity
            out.append(
                AircraftState(
                    self.icao24,
                    float(t[k]),
                    float(lat[k]),
                    float(lon[k]),
                    float(alt[k]),
                    True,
                    self.callsign,
                    float(np.hypot(e, n)),
                    float(np.degrees(np.arctan2(e, n)) % 360),
                    float(u),
                )
            )
        return out


def flight_through(
    observer: Observer,
    direction_enu: np.ndarray,
    when: datetime,
    altitude_m: float,
    speed_m_s: float,
    heading_deg: float,
    icao24: str,
    callsign: str,
    vertical_m_s: float = 0.0,
) -> SimFlight:
    """A flight that is seen in (geometric) ``direction_enu`` at ``when``, at
    ``altitude_m`` above the ellipsoid, flying level towards ``heading_deg``."""
    from ..aircraft import ecef_to_geodetic, enu_basis, geodetic_to_ecef

    origin = geodetic_to_ecef(observer.latitude, observer.longitude, observer.elevation_m)
    d = np.asarray(direction_enu, dtype=float) @ enu_basis(observer.latitude, observer.longitude)
    lo, hi = 0.0, 2_000_000.0  # slant range (m) where the line of sight reaches the altitude
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ecef_to_geodetic(origin + mid * d)[2] < altitude_m else (lo, mid)
    xyz = origin + lo * d
    lat, lon, _ = ecef_to_geodetic(xyz)
    east, north, up = enu_basis(float(lat), float(lon))
    h = np.radians(heading_deg)
    velocity = speed_m_s * (np.sin(h) * east + np.cos(h) * north) + vertical_m_s * up
    return SimFlight(icao24, callsign, when, xyz, velocity)


def render_aircraft_video(
    out_path: str | Path,
    camera: CameraModel,
    observer: Observer,
    start: datetime,
    seconds: float,
    fps: float,
    stars: StarCatalog,
    flights: list[SimFlight],
    light_amp: float = 70.0,
    strobe_hz: float = 1.0,
    noise: float = 2.5,
    seed: int = 0,
    visible: list[tuple[float, float] | None] | None = None,
) -> list[SimFlight]:
    """Render aircraft over a drifting real star field: a steady navigation light and a
    white anti-collision strobe (``strobe_hz`` flashes a second) where each flight really
    is, seen from ``observer`` (apparent, refracted directions). ``visible`` optionally
    limits each flight to (from_s, to_s) of the video, as if it flew behind a cloud."""
    from ..aircraft import enu_basis, geodetic_to_ecef

    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    origin = geodetic_to_ecef(observer.latitude, observer.longitude, observer.elevation_m)
    to_enu = enu_basis(observer.latitude, observer.longitude)
    layers: dict[int, np.ndarray] = {}
    flash_frames = max(1, round(0.1 * fps))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (camera.width, camera.height)
    )
    n = int(seconds * fps)
    for k in range(n):
        block = int(k / fps // 2)
        if block not in layers:
            when = start + timedelta(seconds=(block + 0.5) * 2)
            layers[block] = render_starfield(camera, observer, when, stars, noise=0.0).astype(
                np.float32
            )
        img = layers[block] + rng.normal(0, noise, layers[block].shape).astype(np.float32)
        t = start.timestamp() + k / fps
        for j, f in enumerate(flights):
            window = visible[j] if visible is not None else None
            if window is not None and not window[0] <= k / fps <= window[1]:
                continue
            enu = (f.position(np.array([t]))[0] - origin) @ to_enu.T
            if not np.isfinite(enu).all():  # no position known at this moment
                continue
            enu /= np.linalg.norm(enu)
            if enu[2] <= 0:
                continue
            uv, ok = camera.vec_to_pixel(apply_refraction(enu[None]))
            if not camera.in_image(uv, ok, margin=4)[0]:
                continue
            x, y = float(uv[0, 0]), float(uv[0, 1])
            _stamp(img, x, y, light_amp * 0.6, 1.4)
            if (k + 7 * j) % max(1, round(fps / strobe_hz)) < flash_frames:
                _stamp(img, x + 3, y, light_amp * 1.6, 1.6)
            f.points.append((k, x, y))
        writer.write(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    writer.release()
    return flights
