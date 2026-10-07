"""Camera model: pixel coordinates ↔ unit direction vectors.

A rectilinear (pinhole) projection with Brown radial distortion:

    camera frame: x → image right, y → image down, z → boresight
    (x, y) = (cx_ray/cz_ray, cy_ray/cz_ray)          # gnomonic projection
    r² = x² + y²;  d = 1 + k1·r² + k2·r⁴             # radial distortion
    u = cx + f·d·x,   v = cy + f·d·y                  # pixels

``rotation`` maps camera-frame vectors into the reference frame (ENU for a calibrated
station). Fitting it from matched stars is Wahba's problem, the same attitude-determination
problem a spacecraft star tracker solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .frames import vec_to_altaz


@dataclass
class CameraModel:
    width: int
    height: int
    f: float  # focal length in pixels
    cx: float
    cy: float
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # camera → reference
    k1: float = 0.0
    k2: float = 0.0

    # ----------------------------------------------------------------- projection
    def _distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = x * x + y * y
        d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return x * d, y * d

    def _undistort(self, xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = xd.copy(), yd.copy()
        for _ in range(20):  # fixed-point iteration; converges fast for moderate distortion
            r2 = x * x + y * y
            d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            x, y = xd / d, yd / d
        return x, y

    def pixel_to_vec(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixel coordinates → (N, 3) unit vectors in the reference frame."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        x, y = self._undistort((uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f)
        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays @ self.rotation.T

    def _max_valid_radius(self) -> float:
        """Undistorted radius (normalised) a bit beyond the farthest image corner.

        The distortion polynomial is only fitted inside the image; far outside it can bend
        back and map off-screen directions *into* the frame, so projections beyond this
        radius are treated as not visible.
        """
        corners = np.array(
            [[0.0, 0.0], [self.width, 0.0], [0.0, self.height], [self.width, self.height]]
        )
        x, y = self._undistort(
            (corners[:, 0] - self.cx) / self.f, (corners[:, 1] - self.cy) / self.f
        )
        return float(np.hypot(x, y).max() * 1.15)

    def vec_to_pixel(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) reference-frame vectors → ((N, 2) pixels, (N,) bool "projectable").

        The mask is False for directions behind the camera or far outside the field of view.
        """
        cam = np.atleast_2d(np.asarray(v, dtype=float)) @ self.rotation
        front = cam[:, 2] > 1e-6
        z = np.where(front, cam[:, 2], 1.0)
        x, y = cam[:, 0] / z, cam[:, 1] / z
        valid = front & (x * x + y * y <= self._max_valid_radius() ** 2)
        xd, yd = self._distort(x, y)
        return np.stack([self.cx + self.f * xd, self.cy + self.f * yd], axis=1), valid

    def in_image(self, uv: np.ndarray, front: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (
            front
            & (uv[:, 0] >= margin)
            & (uv[:, 0] < self.width - margin)
            & (uv[:, 1] >= margin)
            & (uv[:, 1] < self.height - margin)
        )

    def distortion_is_physical(self) -> bool:
        """True if the radial mapping r → r·d(r) keeps increasing out to the image corners.

        A real lens never maps two different field angles to the same image radius; a
        fitted model that "folds" is an overfit to wrong matches.
        """
        corner = (
            max(
                np.hypot(px - self.cx, py - self.cy)
                for px in (0, self.width)
                for py in (0, self.height)
            )
            / self.f
        )
        r = np.linspace(0.0, corner * 1.05, 200)
        rd = r * (1.0 + self.k1 * r**2 + self.k2 * r**4)
        return bool(np.all(np.diff(rd) > 0))

    # ----------------------------------------------------------------- descriptors
    @property
    def boresight(self) -> np.ndarray:
        """Unit vector of the image centre direction."""
        return self.pixel_to_vec(np.array([[self.width / 2, self.height / 2]]))[0]

    @property
    def fov_deg(self) -> tuple[float, float]:
        """Horizontal and vertical field of view, measured through the image centre."""
        w, h = self.width, self.height
        a = self.pixel_to_vec(np.array([[0.0, h / 2], [w, h / 2], [w / 2, 0.0], [w / 2, h]]))
        return (
            float(np.degrees(np.arccos(np.clip(a[0] @ a[1], -1, 1)))),
            float(np.degrees(np.arccos(np.clip(a[2] @ a[3], -1, 1)))),
        )

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(np.degrees(1.0 / self.f) * 3600.0)

    def pointing_altaz(self) -> dict[str, float]:
        """For an ENU-referenced model: where the camera looks and how it is rotated.

        ``roll`` is the camera's rotation about its line of sight: 0 = horizon level in the
        image; positive = camera turned clockwise as seen from behind it (the right side of
        the image dips toward the horizon).
        """
        alt, az = vec_to_altaz(self.boresight)
        b = self.boresight
        up_img = self.rotation @ np.array([0.0, -1.0, 0.0])  # image "up" direction
        zenith = np.array([0.0, 0.0, 1.0])
        ref = zenith - (zenith @ b) * b  # zenith projected onto the image plane
        if np.linalg.norm(ref) < 1e-9:  # looking straight up: use north instead
            ref = np.array([0.0, 1.0, 0.0]) - b[1] * b
        ref /= np.linalg.norm(ref)
        roll = np.degrees(np.arctan2(np.cross(ref, up_img) @ b, ref @ up_img))
        return {"alt": float(alt), "az": float(az), "roll": float(roll)}

    # ----------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "rectilinear-radial",
            "width": self.width,
            "height": self.height,
            "f": self.f,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "rotation": self.rotation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraModel:
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            f=float(d["f"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            k1=float(d["k1"]),
            k2=float(d["k2"]),
            rotation=np.array(d["rotation"], dtype=float),
        )


def wahba_rotation(
    cam_vecs: np.ndarray, ref_vecs: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Best rotation R with ref ≈ R @ cam (SVD solution to Wahba's problem)."""
    w = np.ones(len(cam_vecs)) if weights is None else weights
    B = (ref_vecs * w[:, None]).T @ cam_vecs
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


@dataclass
class FitResult:
    model: CameraModel
    residuals_px: np.ndarray  # per-star reprojection error
    inliers: np.ndarray  # bool mask

    @property
    def rms_px(self) -> float:
        r = self.residuals_px[self.inliers]
        return float(np.sqrt(np.mean(r * r))) if r.size else float("nan")


def fit_camera(
    pixels: np.ndarray,
    ref_vecs: np.ndarray,
    width: int,
    height: int,
    f_init: float = 1000.0,
    fit_distortion: bool = True,
    fit_center: bool = True,
    outlier_px: float = 3.0,
    init: CameraModel | None = None,
) -> FitResult:
    """Fit a CameraModel to matched (pixel, direction) pairs.

    Starts from a pinhole guess (rotation via Wahba's SVD solution), then refines rotation,
    focal length, optical centre and distortion with robust least squares, discarding
    matches whose reprojection error stays above ``outlier_px``.
    """
    pixels = np.asarray(pixels, float)
    ref_vecs = np.asarray(ref_vecs, float)
    if init is not None:
        # Continue from a previous solution (keeps its distortion and centre as a start).
        model = CameraModel(
            width, height, init.f, init.cx, init.cy, init.rotation.copy(), init.k1, init.k2
        )
    else:
        model = CameraModel(width, height, f_init, width / 2, height / 2)
        model.rotation = wahba_rotation(_rays(pixels, model), ref_vecs)

    n_free = 4 + (2 if fit_center else 0) + (2 if fit_distortion else 0)
    inliers = np.ones(len(pixels), dtype=bool)
    if len(pixels) < max(4, n_free // 2 + 1):
        # Too few stars to fit everything: keep the pinhole solution.
        fit_distortion = fit_center = False

    def unpack(p: np.ndarray, base: CameraModel) -> CameraModel:
        rot = base.rotation @ Rotation.from_rotvec(p[:3]).as_matrix()
        i = 4
        cx, cy = (p[i], p[i + 1]) if fit_center else (base.cx, base.cy)
        i += 2 if fit_center else 0
        k1, k2 = (p[i], p[i + 1]) if fit_distortion else (base.k1, base.k2)
        return CameraModel(width, height, p[3], cx, cy, rot, k1, k2)

    for _ in range(3):
        base = model
        p0 = [0.0, 0.0, 0.0, model.f]
        # Typical parameter sizes, so the optimiser treats radians, pixels and distortion
        # coefficients on comparable footing.
        scale = [1e-3, 1e-3, 1e-3, 1.0]
        if fit_center:
            p0 += [model.cx, model.cy]
            scale += [1.0, 1.0]
        if fit_distortion:
            p0 += [model.k1, model.k2]
            scale += [1e-3, 1e-3]
        sel_px, sel_vec = pixels[inliers], ref_vecs[inliers]

        def residuals(p: np.ndarray, sel_px=sel_px, sel_vec=sel_vec, base=base) -> np.ndarray:
            uv, front = unpack(p, base).vec_to_pixel(sel_vec)
            r = uv - sel_px
            r[~front] = 1e3
            return r.ravel()

        sol = least_squares(
            residuals, np.array(p0, float), loss="soft_l1", f_scale=1.0, x_scale=np.array(scale)
        )
        model = unpack(sol.x, base)
        uv, front = model.vec_to_pixel(ref_vecs)
        err = np.where(front, np.linalg.norm(uv - pixels, axis=1), np.inf)
        new_inliers = err < outlier_px
        if new_inliers.sum() < 4 or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers

    uv, front = model.vec_to_pixel(ref_vecs)
    err = np.where(front, np.linalg.norm(uv - pixels, axis=1), np.inf)
    return FitResult(model, err, err < outlier_px)


def _rays(pixels: np.ndarray, model: CameraModel) -> np.ndarray:
    """Camera-frame unit rays for pixels (no rotation applied)."""
    x = (pixels[:, 0] - model.cx) / model.f
    y = (pixels[:, 1] - model.cy) / model.f
    r = np.stack([x, y, np.ones_like(x)], axis=1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)
