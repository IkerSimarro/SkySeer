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
        area_factor = self.scale**2
        self._min_area = config.min_area * area_factor
        self._max_area = config.max_area_fraction * width * height * area_factor
        self.last_mask: np.ndarray | None = None

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        gray_full = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = (
            gray_full
            if self.scale == 1.0
            else cv2.resize(gray_full, self._work_size, interpolation=cv2.INTER_AREA)
        )
        k = self.config.blur_kernel
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        mask = self._subtractor.apply(blurred)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        self.last_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / self.scale
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self._min_area < area < self._max_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            m = cv2.moments(contour)
            if m["m00"]:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                cx, cy = x + w / 2, y + h / 2

            # Map back to full resolution for brightness and reporting.
            fx, fy = int(x * inv), int(y * inv)
            fw, fh = max(1, int(round(w * inv))), max(1, int(round(h * inv)))
            roi = gray_full[fy : fy + fh, fx : fx + fw]
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
                )
            )
        return detections
