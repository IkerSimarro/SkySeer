"""Astrometric calibration: find exactly where the camera points from the stars it sees.

Steps:
  1. Stack a few frames to suppress noise and moving objects, and extract star centroids.
  2. Lost-in-space plate solve (tetra3): match the star pattern against the catalogue.
     This needs no location, time or pointing hint.
  3. Convert matched stars to the observer's horizon frame (ENU) at the frame time,
     including atmospheric refraction, and fit the camera model (attitude, focal length,
     optical centre, lens distortion).
  4. Iteratively project every catalogue star into the image, match more stars out to the
     corners, and refit. Wide lenses distort most at the edges, so this step matters.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ._vendor import tetra3
from .camera import CameraModel, fit_camera
from .catalog import StarCatalog, build_database, load_stars
from .frames import (
    Observer,
    apply_refraction,
    icrs_to_enu_matrices,
    icrs_to_enu_matrix,
    radec_to_vec,
    remove_refraction,
    sidereal_drift,
    vec_to_altaz,
    vec_to_radec,
)

log = logging.getLogger(__name__)


class CalibrationError(RuntimeError):
    pass


@dataclass
class Calibration:
    """A solved camera.

    With an observer it is referenced to the local horizon (ENU) and stays valid for as long
    as the camera isn't moved. Without one it is referenced to the stars (ICRS) at ``time``;
    sky coordinates at other times are still exact (the sky's rotation is applied), only
    altitude/azimuth and refraction are unavailable.
    """

    camera: CameraModel
    time: datetime
    observer: Observer | None
    n_stars: int
    rms_px: float
    solver: dict[str, Any] = field(default_factory=dict)
    matched: list[dict[str, float]] = field(default_factory=list)  # for overlays/debugging

    @property
    def frame(self) -> str:
        return "enu" if self.observer else "icrs"

    @property
    def rms_arcsec(self) -> float:
        return self.rms_px * self.camera.pixel_scale_arcsec

    def pixel_to_radec(
        self, uv: np.ndarray, when: datetime | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        v = self.camera.pixel_to_vec(uv)
        if self.observer is not None:
            v = remove_refraction(v)
            v = v @ icrs_to_enu_matrix(when or self.time, self.observer)  # ENU → ICRS
        elif when is not None and when != self.time:
            v = v @ sidereal_drift(self.time, when).T
        return vec_to_radec(v)

    def pixel_to_altaz(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apparent (refracted) altitude and azimuth of pixels, in degrees."""
        if self.observer is None:
            raise CalibrationError("Altitude/azimuth needs the observer location")
        return vec_to_altaz(self.camera.pixel_to_vec(uv))

    def radec_to_pixel(
        self, ra: np.ndarray, dec: np.ndarray, when: datetime | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        v = radec_to_vec(ra, dec)
        if self.observer is not None:
            v = apply_refraction(v @ icrs_to_enu_matrix(when or self.time, self.observer).T)
        elif when is not None and when != self.time:
            v = v @ sidereal_drift(self.time, when)
        return self.camera.vec_to_pixel(v)

    def pixel_to_radec_at(
        self, uv: np.ndarray, times: list[datetime]
    ) -> tuple[np.ndarray, np.ndarray]:
        """RA/Dec of each pixel at its own timestamp (vectorised over a track)."""
        v = self.camera.pixel_to_vec(uv)
        ref = self.observer if self.observer is not None else Observer(0.0, 0.0)
        mats = icrs_to_enu_matrices(times, ref)  # (N, 3, 3) ICRS → ENU
        if self.observer is not None:
            v = remove_refraction(v)
        else:
            # Pixel direction is fixed on the ground: ICRS(t0) → ground → ICRS(t).
            v = v @ icrs_to_enu_matrix(self.time, ref).T
        v_icrs = np.einsum("nji,nj->ni", mats, v)  # apply Mᵀ per point
        return vec_to_radec(v_icrs)

    def pixel_to_enu(self, uv: np.ndarray) -> np.ndarray:
        """Apparent (refracted) ENU unit vectors for pixels. Needs the observer location."""
        if self.observer is None:
            raise CalibrationError("Horizon directions need the observer location")
        return self.camera.pixel_to_vec(uv)

    def summary(self) -> dict[str, Any]:
        hfov, vfov = self.camera.fov_deg
        centre = np.array([[self.camera.width / 2, self.camera.height / 2]])
        ra, dec = self.pixel_to_radec(centre)
        out: dict[str, Any] = {
            "time": self.time.isoformat(),
            "center_ra": float(ra[0]),
            "center_dec": float(dec[0]),
            "fov_h": hfov,
            "fov_v": vfov,
            "pixel_scale_arcsec": self.camera.pixel_scale_arcsec,
            "distortion_k1": self.camera.k1,
            "distortion_k2": self.camera.k2,
            "stars_matched": self.n_stars,
            "rms_px": self.rms_px,
            "rms_arcsec": self.rms_arcsec,
        }
        if self.observer is not None:
            out.update({f"pointing_{k}": v for k, v in self.camera.pointing_altaz().items()})
        return out

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "time": self.time.isoformat(),
            "observer": None
            if self.observer is None
            else {
                "latitude": self.observer.latitude,
                "longitude": self.observer.longitude,
                "elevation_m": self.observer.elevation_m,
            },
            "camera": self.camera.to_dict(),
            "n_stars": self.n_stars,
            "rms_px": self.rms_px,
            "solver": self.solver,
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        obs = Observer(**d["observer"]) if d["observer"] else None
        return cls(
            CameraModel.from_dict(d["camera"]),
            datetime.fromisoformat(d["time"]),
            obs,
            d["n_stars"],
            d["rms_px"],
            d.get("solver", {}),
        )


# ---------------------------------------------------------------------- image prep
def stack_frames(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Median-combine grayscale frames: noise drops and moving objects disappear."""
    gray = [f if f.ndim == 2 else f.mean(axis=2) for f in frames]
    if not gray:
        raise CalibrationError("No frames to stack")
    return np.median(np.stack(gray).astype(np.float32), axis=0)


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def load_sky_image(
    path: str | Path,
    start_time: datetime | None = None,
    at_seconds: float = 0.0,
    seconds: float = 2.0,
    max_frames: int = 30,
) -> tuple[np.ndarray, datetime]:
    """A clean grayscale star image and its UTC time, from a video (stacked) or a photo."""
    import cv2

    from ..sources import FileSource

    path = Path(path)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        source = FileSource(path, start_time)
        first = int(at_seconds * source.fps)
        last = first + max(1, int(seconds * source.fps))
        step = max(1, (last - first) // max_frames)
        frames, times = [], []
        try:
            for frame in source:
                if frame.index >= last:
                    break
                if frame.index >= first and (frame.index - first) % step == 0:
                    frames.append(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY))
                    times.append(frame.timestamp)
        finally:
            source.close()
        if not frames:
            raise CalibrationError(f"No frames read from {path} at {at_seconds}s")
        return stack_frames(frames), times[len(times) // 2]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CalibrationError(f"Could not read image {path}")
    if start_time is None:
        start_time = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        log.warning("No time given; using the file's modification time %s", start_time)
    return image.astype(np.float32), start_time.astimezone(UTC)


def sky_mask(image: np.ndarray, margin: int = 10) -> np.ndarray:
    """Boolean mask of open sky, excluding trees, roofs and other dark foreground.

    At night the sky glows faintly while silhouettes are darker, so a heavily smoothed
    image separates the two. The boundary is pushed ``margin`` px into the sky, because
    edges of silhouettes produce spurious star-like peaks. Returns all-True when there is
    no clear foreground (e.g. an all-sky camera).
    """
    import cv2

    img = np.asarray(image, dtype=np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), 12)
    p5, p50 = np.percentile(blur, [5, 50])
    if p50 - p5 < 1.0:
        return np.ones(img.shape, dtype=bool)
    sky = (blur > p5 + 0.5 * (p50 - p5)).astype(np.uint8)
    k = 2 * margin + 1
    return cv2.erode(sky, np.ones((k, k), np.uint8)).astype(bool)


def find_stars(
    image: np.ndarray, max_stars: int = 400, mask: np.ndarray | None = None, k_sigma: float = 5.0
) -> np.ndarray:
    """Star centroids as (N, 2) pixel (x, y), brightest first.

    Designed for real video as much as clean astro images: the local background is removed,
    noise is measured robustly over the sky, and only compact peaks are kept, which
    rejects silhouette edges, cloud texture and compression blocks. Pixel convention
    matches the motion detector: integer coordinates are pixel centres.
    """
    import cv2

    img = np.asarray(image, dtype=np.float32)
    if mask is None:
        mask = sky_mask(img)
    resid = img - cv2.GaussianBlur(img, (0, 0), 4)
    vals = resid[mask]
    if vals.size < 100:
        return np.zeros((0, 2))
    sigma = 1.4826 * float(np.median(np.abs(vals - np.median(vals)))) + 1e-6
    is_peak = (resid == cv2.dilate(resid, np.ones((3, 3), np.uint8))) & (resid > k_sigma * sigma)
    is_peak &= mask
    is_peak[:3, :] = is_peak[-3:, :] = False
    is_peak[:, :3] = is_peak[:, -3:] = False
    ys, xs = np.nonzero(is_peak)
    pos = np.clip(resid, 0, None)
    yy, xx = np.mgrid[-1:2, -1:2]
    found: list[tuple[float, float, float]] = []
    for y, x in zip(ys, xs, strict=True):
        w3 = pos[y - 1 : y + 2, x - 1 : x + 2]
        f3, f7 = float(w3.sum()), float(pos[y - 3 : y + 4, x - 3 : x + 4].sum())
        if f7 <= 0 or f3 / f7 < 0.55:  # light spread out: an edge, ridge or cloud, not a star
            continue
        found.append((f3, x + float((xx * w3).sum()) / f3, y + float((yy * w3).sum()) / f3))
    found.sort(reverse=True)
    return np.array([(x, y) for _, x, y in found[:max_stars]]).reshape(-1, 2)


# ---------------------------------------------------------------------- solving
_SOLVERS: dict[Path, tetra3.Tetra3] = {}


def _solver(db: Path) -> tetra3.Tetra3:
    if db not in _SOLVERS:
        _SOLVERS[db] = tetra3.Tetra3(db)
    return _SOLVERS[db]


def _match(proj: np.ndarray, stars: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Mutual-nearest-neighbour matches between projected catalogue stars and centroids."""
    if len(proj) == 0 or len(stars) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d1, j = cKDTree(stars).query(proj, distance_upper_bound=radius)
    _, i_back = cKDTree(proj).query(stars)
    ok = np.isfinite(d1)
    i = np.nonzero(ok)[0]
    j = j[ok]
    mutual = i_back[j] == i
    return i[mutual], j[mutual]


def calibrate(
    image: np.ndarray,
    time: datetime,
    observer: Observer | None = None,
    fov_estimate: float | None = None,
    fov_max_error: float | None = None,
    database: Path | None = None,
    catalog: StarCatalog | None = None,
    refine_mag: float = 6.5,
    min_stars: int = 10,
    max_rms_px: float = 1.5,
    hint: Calibration | None = None,
) -> Calibration:
    """Solve the camera from one (preferably stacked) grayscale image taken at ``time``.

    With a ``hint`` (a previous calibration of the same camera) the slow pattern search is
    skipped: the previous model is refined directly against the catalogue, which takes a
    fraction of a second. If that fails, e.g. because the camera was moved, it falls back to
    a full lost-in-space solve.
    """
    time = time.astimezone(UTC)
    h, w = image.shape[:2]
    stars = find_stars(image)
    log.info("Found %d star candidates", len(stars))
    if len(stars) < 6:
        raise CalibrationError(f"Only {len(stars)} stars found; is the image dark and in focus?")

    to_frame = icrs_to_enu_matrix(time, observer) if observer is not None else np.eye(3)

    def frame_vecs(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        v = radec_to_vec(ra, dec) @ to_frame.T
        return apply_refraction(v) if observer is not None else v

    cat = catalog or load_stars(refine_mag, time.year + time.timetuple().tm_yday / 365.25)
    cat = cat.brighter_than(refine_mag)
    limits = (min_stars, max_rms_px)

    # A hint is only meaningful for the same camera at the same place.
    if (
        hint is not None
        and (hint.camera.width, hint.camera.height) == (w, h)
        and (hint.observer == observer)
    ):
        start = _carry_forward(hint, time, observer)
        try:
            cal = _refine(
                start, stars, cat, frame_vecs, time, observer, {"method": "hint"}, *limits
            )
            log.info("Re-calibrated from previous solution (%d stars)", cal.n_stars)
            return cal
        except CalibrationError as e:
            log.info("Quick re-calibration failed (%s); doing a full solve", e)

    model, solver_info = _lost_in_space(
        stars, w, h, frame_vecs, fov_estimate, fov_max_error, database
    )
    return _refine(model, stars, cat, frame_vecs, time, observer, solver_info, *limits)


def _carry_forward(hint: Calibration, time: datetime, observer: Observer | None) -> CameraModel:
    """The previous camera model, valid at ``time``.

    A horizon-referenced (ENU) model does not change while the camera stays put. A
    star-referenced (ICRS) model must turn with the sky between the two times.
    """
    cam = hint.camera
    if observer is not None:
        return cam
    drift = sidereal_drift(hint.time, time)
    return CameraModel(
        cam.width, cam.height, cam.f, cam.cx, cam.cy, drift @ cam.rotation, cam.k1, cam.k2
    )


def _lost_in_space(
    stars: np.ndarray,
    w: int,
    h: int,
    frame_vecs: Any,
    fov_estimate: float | None,
    fov_max_error: float | None,
    database: Path | None,
) -> tuple[CameraModel, dict[str, Any]]:
    """Pattern-match the brightest stars against the database (no prior knowledge)."""
    solver = _solver(database or build_database())
    t3_stars = stars[:, ::-1] + 0.5
    # Escalating attempts: the fast default usually works; in wide, distorted fields the
    # brightest few stars sometimes fail to form a catalogued pattern, so widen the search.
    attempts: list[dict[str, Any]] = [
        {"pattern_checking_stars": 8, "match_radius": 0.01, "distortion": (-0.3, 0.1)},
        {"pattern_checking_stars": 14, "match_radius": 0.02, "distortion": (-0.3, 0.1)},
        {"pattern_checking_stars": 14, "match_radius": 0.02, "distortion": 0.0},
        {
            "pattern_checking_stars": 20,
            "match_radius": 0.03,
            "distortion": (-0.3, 0.1),
            "fov_estimate": None,
            "fov_max_error": None,
        },
    ]
    result: dict[str, Any] = {}
    for attempt in attempts:
        kwargs: dict[str, Any] = {
            "fov_estimate": fov_estimate,
            "fov_max_error": fov_max_error,
            "solve_timeout": 30_000,
            **attempt,
        }
        result = solver.solve_from_centroids(t3_stars, (h, w), return_matches=True, **kwargs)
        if result.get("RA") is not None:
            break
        log.info("Solve attempt failed (%s); widening search", attempt)
    if result.get("RA") is None:
        raise CalibrationError(
            f"Plate solve failed with {len(stars)} stars (FOV estimate {fov_estimate}). "
            "Check focus, exposure and the field-of-view setting."
        )
    solver_info: dict[str, Any] = {
        k: (float(result[k]) if result[k] is not None else None)
        for k in ("RA", "Dec", "Roll", "FOV", "distortion", "RMSE", "Prob", "T_solve")
    }
    solver_info["Matches"] = int(result["Matches"])
    solver_info["method"] = "lost-in-space"
    log.info(
        "Plate solve: RA %.3f Dec %.3f FOV %.2f, %d stars, %.0f ms",
        result["RA"],
        result["Dec"],
        result["FOV"],
        result["Matches"],
        result["T_solve"],
    )

    m_stars = np.asarray(result["matched_stars"])
    m_px = np.asarray(result["matched_centroids"])[:, ::-1] - 0.5
    f_init = (w / 2) / np.tan(np.radians(result["FOV"]) / 2)
    fit = fit_camera(
        m_px,
        frame_vecs(m_stars[:, 0], m_stars[:, 1]),
        w,
        h,
        f_init,
        fit_distortion=len(m_px) >= 10,
        fit_center=False,
    )
    return fit.model, solver_info


def _refine(
    model: CameraModel,
    stars: np.ndarray,
    cat: StarCatalog,
    frame_vecs: Any,
    time: datetime,
    observer: Observer | None,
    solver_info: dict[str, Any],
    min_stars: int,
    max_rms_px: float,
) -> Calibration:
    """Match the full catalogue across the whole frame, refit, and validate the result."""
    w, h = model.width, model.height
    fit = None
    cat_vecs = frame_vecs(cat.ra, cat.dec)
    if observer is not None:
        above = cat_vecs[:, 2] > np.sin(np.radians(2.0))  # skip stars lost in horizon haze
        cat, cat_vecs = (
            StarCatalog(cat.hip[above], cat.ra[above], cat.dec[above], cat.mag[above]),
            cat_vecs[above],
        )
    radius = 6.0
    fit_i = fit_j = None  # catalogue / centroid indices used by the latest refined fit
    for _ in range(4):
        uv, front = model.vec_to_pixel(cat_vecs)
        vis = np.nonzero(model.in_image(uv, front, margin=2))[0]
        i, j = _match(uv[vis], stars, radius)
        i = vis[i]
        if len(i) < 6:
            break
        fit = fit_camera(
            stars[j],
            cat_vecs[i],
            w,
            h,
            init=model,
            fit_distortion=len(i) >= 12,
            fit_center=len(i) >= 40,
        )
        model, fit_i, fit_j = fit.model, i, j
        log.debug(
            "Refine: %d visible, %d matched (r=%.1f px), %d inliers, RMS %.2f px",
            len(vis),
            len(i),
            radius,
            fit.inliers.sum(),
            fit.rms_px,
        )
        radius = max(2.0, 3.0 * fit.rms_px)

    if fit is None:
        raise CalibrationError("Solution rejected: too few catalogue stars line up")
    matched: list[dict[str, float]] = []
    if fit_i is not None and fit_j is not None:
        for a, b, e, ok in zip(fit_i, fit_j, fit.residuals_px, fit.inliers, strict=True):
            if ok:
                matched.append(
                    {
                        "hip": int(cat.hip[a]),
                        "mag": float(cat.mag[a]),
                        "x": float(stars[b, 0]),
                        "y": float(stars[b, 1]),
                        "err_px": float(e),
                    }
                )
    # Significance: how many matches would pure coincidence give? Each catalogue star
    # projected into the frame has probability (centroid density × match area) of landing
    # on some unrelated detection. A real solution beats that expectation by a wide margin.
    uv, front = model.vec_to_pixel(cat_vecs)
    n_visible = int(model.in_image(uv, front, margin=2).sum())
    match_area = np.pi * radius**2
    chance = n_visible * min(1.0, len(stars) * match_area / (w * h))
    solver_info["visible_catalog_stars"] = n_visible
    solver_info["chance_matches"] = float(chance)

    cal = Calibration(
        model, time, observer, int(fit.inliers.sum()), fit.rms_px, solver_info, matched
    )
    # Independent checks on the solver: a wrong pattern match cannot line up many catalogue
    # stars with small residuals, far beyond chance, through a physically possible lens.
    problems = []
    needed = max(min_stars, int(np.ceil(3 * chance + 10)))
    if cal.n_stars < needed:
        problems.append(
            f"{cal.n_stars} stars matched but chance alone gives ~{chance:.1f} (need {needed})"
        )
    if not cal.rms_px < max_rms_px:
        problems.append(f"residual {cal.rms_px:.2f} px (limit {max_rms_px})")
    if not model.distortion_is_physical():
        problems.append(f"impossible lens distortion (k1={model.k1:.3f}, k2={model.k2:.3f})")
    if problems:
        raise CalibrationError("Solution rejected: " + "; ".join(problems))
    log.info(
        "Calibrated with %d stars, RMS %.2f px (%.0f arcsec)",
        cal.n_stars,
        cal.rms_px,
        cal.rms_arcsec,
    )
    return cal
