"""Photometry: turn pixel brightness into astronomical magnitudes using the stars.

Magnitudes are logarithmic: m = ZP - 2.5·log10(flux). The zero point ZP is measured from
catalogue stars in the calibration image; afterwards any detection's flux converts to a
real magnitude (e.g. Venus about -4, the faintest naked-eye stars about +6).

Simplifications, acceptable for video meteor work (typically ±0.3-0.5 mag): no
colour correction and no atmospheric-extinction term. Saturated detections only give an
upper limit (the object was at least that bright).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def aperture_flux(
    image: np.ndarray, x: float, y: float, radius: float = 4.0
) -> tuple[float, float]:
    """Background-subtracted flux in a circular aperture and the peak pixel value.

    The background is the median of an annulus 2-4 aperture radii out.
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    r_out = int(np.ceil(radius * 4))
    x0, x1 = max(0, int(x) - r_out), min(w, int(x) + r_out + 1)
    y0, y1 = max(0, int(y) - r_out), min(h, int(y) + r_out + 1)
    patch = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    ann = patch[(d >= 2 * radius) & (d <= 4 * radius)]
    sky = float(np.median(ann)) if ann.size else 0.0
    ap = patch[d <= radius]
    return float(ap.sum() - sky * ap.size), float(ap.max()) if ap.size else 0.0


@dataclass
class ZeroPoint:
    value: float
    scatter: float  # robust spread of the per-star estimates (mag)
    n_stars: int

    def magnitude(self, flux: float) -> float | None:
        return self.value - 2.5 * float(np.log10(flux)) if flux > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "scatter": self.scatter, "n_stars": self.n_stars}


def fit_zero_point(
    image: np.ndarray,
    stars: list[dict[str, float]],
    radius: float = 4.0,
    saturation: float | None = None,
) -> ZeroPoint | None:
    """Zero point from matched catalogue stars (dicts with x, y, mag), ignoring saturated
    and too-faint stars, with robust (median, MAD-clipped) statistics."""
    img = np.asarray(image, dtype=np.float32)
    sat = saturation if saturation is not None else 0.97 * float(img.max())
    estimates = []
    for s in stars:
        flux, peak = aperture_flux(img, s["x"], s["y"], radius)
        if flux <= 0 or peak >= sat:
            continue
        estimates.append(s["mag"] + 2.5 * np.log10(flux))
    if len(estimates) < 5:
        return None
    est = np.array(estimates)
    med = float(np.median(est))
    mad = 1.4826 * float(np.median(np.abs(est - med))) + 1e-3
    keep = np.abs(est - med) < 3 * mad
    zp = float(np.median(est[keep]))
    return ZeroPoint(zp, float(1.4826 * np.median(np.abs(est[keep] - zp))), int(keep.sum()))


@dataclass
class EventPhotometry:
    peak_magnitude: float | None
    saturated: bool  # if True the true peak was brighter than peak_magnitude
    magnitudes: list[float | None]  # per detection (light curve)

    @property
    def peak_label(self) -> str:
        if self.peak_magnitude is None:
            return "-"
        return f"{'<' if self.saturated else ''}{self.peak_magnitude:+.1f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_magnitude": None
            if self.peak_magnitude is None
            else round(self.peak_magnitude, 2),
            "saturated": self.saturated,
            "light_curve": [None if m is None else round(m, 2) for m in self.magnitudes],
        }


def event_photometry(fluxes: list[float], saturated: list[bool], zp: ZeroPoint) -> EventPhotometry:
    """Per-detection magnitudes and the peak. The peak is taken from a 3-sample running
    median of the flux, so a single noise spike can't make an object look brighter."""
    from scipy.ndimage import median_filter

    mags = [zp.magnitude(f) for f in fluxes]
    flux = np.asarray(fluxes, dtype=float)
    if not (flux > 0).any():
        return EventPhotometry(None, False, mags)
    smooth = median_filter(flux, size=3, mode="nearest") if len(flux) >= 3 else flux
    k = int(np.argmax(smooth))
    peak = zp.magnitude(float(smooth[k]))
    return EventPhotometry(peak, bool(any(saturated[max(0, k - 1) : k + 2])), mags)
