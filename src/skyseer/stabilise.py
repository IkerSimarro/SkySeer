"""Camera-shake correction (image stabilisation), the way a star tracker measures motion.

Wind on a mast makes a camera wobble by a pixel or two. To the background model that
looks like every star moving at once, which floods it with false detections (measured
on simulated wobble: 45 false objects in 48 s of video at 2 px, 416 at 4 px). The
stabiliser measures each frame's shift from the stars themselves and shifts the frame
back before detection.

How the shift is measured, as spacecraft star trackers do: pick a few dozen bright,
isolated stars; every frame, find each one's centre to a fraction of a pixel
(centroiding: the brightness-weighted mean position in a small window); the camera's
shift is the *median* of the stars' shifts, so a satellite crossing one star doesn't
disturb it. Measured on simulated wobble: 0.05 px error in 0.7 ms per frame. FFT
phase correlation, the textbook alternative, was 28x slower at full resolution (19 ms,
0.12 px error) and unreliable when downscaled to make it faster. (Testing it also
revealed that OpenCV 5.0's ``phaseCorrelate`` multiplies its first input by the window
in place, so a reused reference image silently decays: copy it if you use it.)

Only the *jitter* is removed: the shift minus its slow trend. The trend uses Holt's
linear smoothing, which estimates a level and its rate of change, so it follows the
steady drift of the stars with the Earth's rotation with little lag (at most 0.55 px,
measured), and that drift is left to the background model, which already follows it.
(A least-squares line through all shifts so far removes even that lag, but early on it
follows the wobble itself, and leftover wobble is what causes false detections.)

A knock that leaves the camera pointing slightly differently shows up as a sudden jump
in the shift. Its size is known exactly, so it is simply kept corrected: the picture
stays aligned with the original view, objects are tracked straight through it and the
star calibration stays valid. Only when the camera really moved, beyond what can be
corrected (``max_shift_px``, or all the tracked stars vanishing at once), is ``moved``
set; the pipeline then handles it as a disturbance. This is more reliable than counting
changed pixels, which missed a 25-pixel knock in a sparse star field.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .config import StabiliseConfig

log = logging.getLogger(__name__)


class Stabiliser:
    def __init__(self, config: StabiliseConfig, fps: float):
        self.config = config
        self.fps = fps
        r = config.window_px
        self._offs = np.arange(-r, r + 1)
        self.reset()

    def reset(self) -> None:
        """Forget the stars and the trend (after a disturbance)."""
        self._ref: np.ndarray | None = None  # reference star positions (N, 2)
        self._ref_offset = np.zeros(2)  # camera offset when the reference was taken
        self._offset = np.zeros(2)  # latest measured offset of the stars
        self._level: np.ndarray | None = None  # Holt smoother: smoothed offset...
        self._rate = np.zeros(2)  # ...and its rate of change per processed frame
        self._smoothed = 0
        self._step_scale = 0.0  # typical frame-to-frame motion (how hard it's shaking)
        self._knocks = np.zeros(2)  # sudden jumps so far, kept corrected
        self._frames_since_ref = 0
        self._retry_in = 0
        self._noise = 1.0
        self.correction = (0.0, 0.0)  # shift applied to the last frame (px)
        self.border = 0  # pixels at each edge invalid after that shift
        self.stars = 0  # stars measured on the last frame
        self.moved = False  # the last frame showed a real camera move, not shake

    def apply(self, gray: np.ndarray, frame_skip: int = 1) -> np.ndarray:
        """Return ``gray`` shifted to cancel the camera's jitter (or unchanged)."""
        self.correction, self.border, self.moved = (0.0, 0.0), 0, False
        tracking = self._ref is not None and self.stars >= 0.8 * len(self._ref)
        previous = self._offset.copy()
        offset = self._measure(gray, frame_skip)
        if offset is None:
            # Nearly all reference stars tracked last frame, most lost now: the view jumped.
            self.moved = tracking and self._ref is None
            return gray
        step = offset - previous
        jump = float(np.hypot(*step))
        if self._step_scale > 0 and jump > max(self.config.step_px, 4 * self._step_scale):
            # A sudden jump, far beyond how the camera has been shaking: a knock. Keep it
            # out of the trend, so it stays corrected instead of slowly being followed.
            self._knocks += step
            log.info("Camera knocked by (%+.1f, %+.1f) px; compensating", *step)
        else:
            self._step_scale += (jump - self._step_scale) * (0.1 if self._step_scale else 1.0)
        # The trend follows only the smooth part of the motion (drift), never the knocks.
        jitter = offset - self._smooth(offset - self._knocks, frame_skip)
        size = float(np.abs(jitter).max())
        if size > self.config.max_shift_px:
            self.moved = True  # too far to correct: the camera really moved
            return gray
        if size < self.config.min_shift_px:
            return gray
        self.correction = (float(jitter[0]), float(jitter[1]))
        self.border = math.ceil(size) + 1
        shift = np.array([[1, 0, -jitter[0]], [0, 1, -jitter[1]]], np.float32)
        h, w = gray.shape[:2]
        return cv2.warpAffine(
            gray, shift, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    # ----------------------------------------------------------------- measurement
    def _measure(self, gray: np.ndarray, frame_skip: int) -> np.ndarray | None:
        cfg = self.config
        if self._ref is None:
            if self._retry_in > 0:
                self._retry_in -= 1
                return None
            stars = self._find_stars(gray)
            if stars is None:
                self._retry_in = max(1, round(self.fps / frame_skip))  # try again in ~1 s
                return None
            self._set_reference(stars, self._offset)  # (re)acquire: continue from last offset
        assert self._ref is not None
        # Coarse-to-fine, as star trackers do: each star's brightest pixel in a wide window
        # around where it was last frame gives a rough shift (the median, robust to a
        # satellite or a brighter neighbour in a window), then sub-pixel centroids refine it.
        last = self._ref + (self._offset - self._ref_offset)
        rough = np.median(self._peaks(gray, last) - last, axis=0)
        pos, ok = self._centroids(gray, last + rough)
        shifts = pos - self._ref
        # Most reference stars must agree on the shift. After a jump beyond the search
        # window the search lands on *other* stars, which disagree: that is "lost", never
        # a confident wrong shift.
        agree = ok & (np.hypot(*(shifts - np.median(shifts[ok], axis=0)).T) < 1.0)
        self.stars = int(agree.sum()) if ok.any() else 0
        if self.stars < max(cfg.min_stars, 0.5 * len(self._ref)):
            log.debug("Only %d stars agree; re-picking reference stars", self.stars)
            self._ref = None
            return None
        offset = self._ref_offset + np.median(shifts[agree], axis=0)
        self._offset = offset
        self._frames_since_ref += 1
        if self._frames_since_ref * frame_skip > cfg.keyframe_s * self.fps:
            # Fresh stars (the sky drifts and some set or fade), measured on this frame, so
            # the chain of offsets continues exactly from this frame's measured offset.
            stars = self._find_stars(gray)
            if stars is not None:
                self._set_reference(stars, offset)
            self._frames_since_ref = 0
        return offset

    def _set_reference(self, stars: np.ndarray, offset: np.ndarray) -> None:
        self._ref, self._ref_offset = stars, offset.copy()
        self._frames_since_ref = 0

    def _find_stars(self, gray: np.ndarray) -> np.ndarray | None:
        """Bright, isolated, unsaturated stars away from the edges."""
        cfg = self.config
        g = gray.astype(np.float32)
        resid = (g - cv2.GaussianBlur(g, (0, 0), 2))[::4, ::4]
        self._noise = max(0.5, 1.4826 * float(np.median(np.abs(resid - np.median(resid)))))
        smooth = cv2.GaussianBlur(g, (0, 0), 1.0)
        height = smooth - cv2.blur(g, (31, 31))
        isolated = smooth == cv2.dilate(smooth, np.ones((9, 9), np.uint8))
        peak = isolated & (height > cfg.min_star_snr * self._noise) & (gray < 240)
        m = cfg.search_px + math.ceil(cfg.max_shift_px) + 3
        peak[:m], peak[-m:], peak[:, :m], peak[:, -m:] = False, False, False, False
        ys, xs = np.nonzero(peak)
        best = np.argsort(-height[ys, xs])[: cfg.max_stars]
        if best.size < cfg.min_stars:
            return None
        pos, ok = self._centroids(gray, np.stack([xs[best], ys[best]], axis=1).astype(float))
        return pos[ok] if ok.sum() >= cfg.min_stars else None

    def _peaks(self, gray: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Brightest pixel within ``search_px`` of each position (the coarse search)."""
        r = self.config.search_px
        offs, h, w = np.arange(-r, r + 1), gray.shape[0], gray.shape[1]
        ix = np.round(pos).astype(int)
        xs = np.clip(ix[:, 0, None, None] + offs[None, None, :], 0, w - 1)
        ys = np.clip(ix[:, 1, None, None] + offs[None, :, None], 0, h - 1)
        brightest = gray[ys, xs].reshape(len(pos), -1).argmax(axis=1)
        dy, dx = np.divmod(brightest, 2 * r + 1)
        return np.stack([ix[:, 0] + offs[dx], ix[:, 1] + offs[dy]], axis=1).astype(float)

    def _centroids(self, gray: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sub-pixel star centres near ``pos`` (two passes, re-centring the window)."""
        offs, h, w = self._offs, gray.shape[0], gray.shape[1]
        ok = np.ones(len(pos), bool)
        for _ in range(2):
            ix = np.round(pos).astype(int)
            xs = np.clip(ix[:, 0, None, None] + offs[None, None, :], 0, w - 1)
            ys = np.clip(ix[:, 1, None, None] + offs[None, :, None], 0, h - 1)
            patch = gray[ys, xs].astype(np.float32)
            sky = np.median(patch.reshape(len(pos), -1), axis=1)
            weight = np.clip(patch - sky[:, None, None] - self._noise, 0, None)
            total = weight.sum(axis=(1, 2))
            ok = (patch.max(axis=(1, 2)) - sky > self.config.min_star_snr * self._noise) & (
                total > 0
            )
            total = np.where(total > 0, total, 1.0)
            pos = np.stack(
                [
                    ix[:, 0] + (weight * offs[None, None, :]).sum(axis=(1, 2)) / total,
                    ix[:, 1] + (weight * offs[None, :, None]).sum(axis=(1, 2)) / total,
                ],
                axis=1,
            )
        return pos, ok

    def _smooth(self, offset: np.ndarray, frame_skip: int) -> np.ndarray:
        """The slow trend of the offset (Holt's linear smoothing).

        For the first second it's a plain running mean: starting the filter from the first
        frame would start it wherever the wobble happened to be at that moment.
        """
        per_s = self.fps / frame_skip
        if self._level is None:
            self._level, self._rate, self._smoothed = offset.copy(), np.zeros(2), 1
            return self._level
        self._smoothed += 1
        if self._smoothed <= per_s:
            self._level = self._level + (offset - self._level) / self._smoothed
            return self._level
        alpha = min(1.0, 2.0 / (self.config.trend_s * per_s))
        previous = self._level
        self._level = alpha * offset + (1 - alpha) * (previous + self._rate)
        self._rate = alpha / 4 * (self._level - previous) + (1 - alpha / 4) * self._rate
        return self._level
