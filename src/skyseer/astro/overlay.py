"""Visual proof of a calibration: matched stars circled and named, plus a coordinate grid."""

from __future__ import annotations

import cv2
import numpy as np

from .calibrate import Calibration
from .catalog import StarCatalog
from .frames import altaz_to_vec, radec_to_vec
from .starnames import STAR_NAMES

_STAR = (120, 255, 120)  # BGR
_NAME = (190, 255, 190)
_GRID = (200, 140, 60)


def _label(
    img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 3, y - th - 4), (x + tw + 3, y + base), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _polyline(
    img: np.ndarray, cal: Calibration, vecs: np.ndarray, color: tuple[int, int, int]
) -> None:
    uv, front = cal.camera.vec_to_pixel(vecs)
    inside = cal.camera.in_image(uv, front, margin=-50)
    pts: list[list[int]] = []
    for p, ok in zip(uv, inside, strict=True):
        if ok:
            pts.append([int(p[0]), int(p[1])])
        elif len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, color, 1, cv2.LINE_AA)
            pts = []
        else:
            pts = []
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts)], False, color, 1, cv2.LINE_AA)


def draw_calibration(
    image: np.ndarray, cal: Calibration, catalog: StarCatalog | None = None
) -> np.ndarray:
    """Return a copy of ``image`` (gray or BGR) annotated with the calibration."""
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    img = np.ascontiguousarray(img)
    scale = max(0.45, img.shape[1] / 2400)

    # Grid: altitude/azimuth when the location is known, otherwise RA/Dec.
    t = np.linspace(0, 360, 721)
    if cal.observer is not None:
        for alt in range(0, 90, 10):
            _polyline(img, cal, altaz_to_vec(np.full_like(t, alt), t), _GRID)
        for az in range(0, 360, 15):
            _polyline(img, cal, altaz_to_vec(np.linspace(0, 90, 181), np.full(181, az)), _GRID)
        for az, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            uv, front = cal.camera.vec_to_pixel(altaz_to_vec(np.array([2.0]), np.array([az])))
            if cal.camera.in_image(uv, front, margin=20)[0]:
                _label(img, name, (int(uv[0, 0]), int(uv[0, 1])), (80, 200, 255), scale * 1.3)
    else:
        # The ICRS-frame model maps sky vectors directly.
        for dec in range(-80, 90, 10):
            _polyline(img, cal, radec_to_vec(t, np.full_like(t, dec)), _GRID)
        for ra in range(0, 360, 15):
            d = np.linspace(-89, 89, 357)
            _polyline(img, cal, radec_to_vec(np.full_like(d, ra), d), _GRID)

    r = max(6, int(10 * scale))
    for m in cal.matched:
        c = (int(round(m["x"])), int(round(m["y"])))
        cv2.circle(img, c, r, _STAR, 1, cv2.LINE_AA)
        star_name = STAR_NAMES.get(int(m["hip"]))
        if star_name:
            _label(img, star_name, (c[0] + r + 3, c[1] - r), _NAME, scale)

    s = cal.summary()
    lines = [
        f"RA {s['center_ra']:.2f}  Dec {s['center_dec']:+.2f}   "
        f"FOV {s['fov_h']:.1f} x {s['fov_v']:.1f} deg",
        f'{s["stars_matched"]} stars matched   residual {s["rms_arcsec"]:.0f}" '
        f"({s['rms_px']:.2f} px)",
    ]
    if cal.observer is not None:
        lines.insert(
            1,
            f"Pointing az {s['pointing_az']:.2f}  alt {s['pointing_alt']:.2f}  "
            f"roll {s['pointing_roll']:+.2f} deg",
        )
    y = int(30 * scale * 2)
    for line in lines:
        _label(img, line, (int(15 * scale * 2), y), (255, 255, 255), scale)
        y += int(26 * scale * 2)
    return img
