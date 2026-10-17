"""Motion detection: background subtraction + blob extraction."""

from __future__ import annotations

import cv2
import numpy as np

from .config import DetectorConfig
from .models import Detection, Frame


class MotionDetector:
    """Finds moving blobs with MOG2 background subtraction.

    Detection can run on a downscaled image (``process_scale``) to save CPU on small
    boards; coordinates, boxes and areas are always reported in full-resolution pixels.
    """

    @staticmethod
    def _saturation_level(image: np.ndarray) -> float:
        """Pixel value at which the sensor/encoding clips (250 for 8-bit video)."""
        if np.issubdtype(image.dtype, np.integer):
            return 0.98 * float(np.iinfo(image.dtype).max)
        return float("inf")

    @staticmethod
    def _sky_noise(gray: np.ndarray) -> float:
        """Robust pixel noise of the frame (high-pass residual, sampled for speed)."""
        g = gray.astype(np.float32)
        resid = (g - cv2.GaussianBlur(g, (0, 0), 2))[::4, ::4]
        return 1.4826 * float(np.median(np.abs(resid - np.median(resid))))

    def reset(self) -> None:
        """Forget the background (after the camera moved or the scene changed)."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.mog2_history,
            varThreshold=self.config.mog2_var_threshold,
            detectShadows=False,
        )

    @property
    def changed_fraction(self) -> float:
        """Fraction of the last frame flagged as changed (0..1)."""
        if self.last_mask is None:
            return 0.0
        return float(np.count_nonzero(self.last_mask)) / self.last_mask.size

    def __init__(self, config: DetectorConfig, frame_size: tuple[int, int]):
        self.config = config
        width, height = frame_size
        self.scale = config.process_scale
        self._work_size = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.mog2_history,
            varThreshold=config.mog2_var_threshold,
            detectShadows=False,
        )
        self._kernel = np.ones((config.morph_kernel, config.morph_kernel), np.uint8)
        g = 2 * config.group_px + 1
        self._group_kernel = np.ones((g, g), np.uint8) if config.group_px > 0 else None
        area_factor = self.scale**2
        self._min_area = config.min_area * area_factor
        self._max_area = config.max_area_fraction * width * height * area_factor
        self.last_mask: np.ndarray | None = None
        self._noise: float | None = None

    def detect(self, frame: Frame, border: int = 0) -> list[Detection]:
        """Moving blobs in ``frame``. ``border`` px at each edge are ignored (after
        stabilisation shifted the picture, the edge strip holds repeated pixels)."""
        image = frame.image
        gray_full = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = (
            gray_full
            if self.scale == 1.0
            else cv2.resize(gray_full, self._work_size, interpolation=cv2.INTER_AREA)
        )
        k = self.config.blur_kernel
        blurred = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray

        mask = self._subtractor.apply(blurred)
        if self.config.morph_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        if border > 0:
            b = int(np.ceil(border * self.scale)) + 1
            mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = 0, 0, 0, 0
        self.last_mask = mask

        # Connected components give the true pixel count of each blob (an outline's
        # enclosed area is 0 for a 1-2 px object, which used to hide faint satellites).
        # Noise can break a faint object into pieces a pixel or two apart, which the tracker
        # would follow as two objects: pieces are grouped on a slightly widened mask, but
        # measured on the real one.
        grouped = mask if self._group_kernel is None else cv2.dilate(mask, self._group_kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        inv = 1.0 / self.scale
        detections: list[Detection] = []
        if n > 1 and (self._noise is None or frame.index % 15 == 0):
            self._noise = self._sky_noise(gray)
        for i in range(1, n):
            gx, gy, gw, gh, _ = (int(v) for v in stats[i])
            group = labels[gy : gy + gh, gx : gx + gw] == i
            group &= mask[gy : gy + gh, gx : gx + gw] > 0
            area = int(np.count_nonzero(group))
            if not (self._min_area <= area < self._max_area):
                continue
            rows, cols = np.flatnonzero(group.any(axis=1)), np.flatnonzero(group.any(axis=0))
            x, y = gx + int(cols[0]), gy + int(rows[0])
            w, h = int(cols[-1] - cols[0]) + 1, int(rows[-1] - rows[0]) + 1
            # Brightness-weighted centroid: sub-pixel positions for the astrometry.
            comp = group[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            roi_w = gray[y : y + h, x : x + w].astype(np.float32)
            weights = np.where(comp, roi_w - roi_w[comp].min() + 1.0, 0.0)
            yy, xx = np.mgrid[0:h, 0:w]
            total = float(weights.sum())
            cx = x + float((xx * weights).sum()) / total
            cy = y + float((yy * weights).sum()) / total

            # Contrast against a ring of sky around the blob (not the whole frame), so
            # strobe flashes stand out even when the sky brightness varies.
            x0, y0 = max(0, x - 3), max(0, y - 3)
            ring = gray[y0 : y + h + 3, x0 : x + w + 3]
            inside = labels[y0 : y + h + 3, x0 : x + w + 3] == i
            around = ring[~inside]
            sky = float(np.median(around)) if around.size else float(ring.min())
            contrast = float(roi_w[comp].max()) - sky
            # Spread of the surroundings relative to this camera's own sky noise: ~1 on a
            # clear sky whatever the camera, several times higher inside cloud texture.
            spread = 1.4826 * float(np.median(np.abs(around - sky))) if around.size else 0.0
            texture = spread / max(self._noise or 0.0, 0.5)

            # Map back to full resolution for brightness and reporting.
            fx, fy = int(x * inv), int(y * inv)
            fw, fh = max(1, int(round(w * inv))), max(1, int(round(h * inv)))
            roi = gray_full[fy : fy + fh, fx : fx + fw]
            # Total brightness above the local sky in a box 2 px around the blob, at full
            # resolution, comparable with the star photometry used for the zero point.
            bx0, by0 = max(0, fx - 2), max(0, fy - 2)
            box = gray_full[by0 : fy + fh + 2, bx0 : fx + fw + 2].astype(np.float32)
            flux = float(box.sum() - sky * box.size) if box.size else 0.0
            peak = float(box.max()) if box.size else 0.0
            detections.append(
                Detection(
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    x=cx * inv,
                    y=cy * inv,
                    bbox=(fx, fy, fw, fh),
                    area=area * inv * inv,
                    mean_brightness=float(roi.mean()) if roi.size else 0.0,
                    max_brightness=float(roi.max()) if roi.size else 0.0,
                    contrast=contrast,
                    texture=texture,
                    flux=flux,
                    saturated=peak >= self._saturation_level(gray_full),
                    snr=contrast / max(self._noise or 0.0, 0.5),
                )
            )
        return detections
