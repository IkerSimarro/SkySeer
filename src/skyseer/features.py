"""Per-track kinematic and photometric features.

``avg_speed`` and friends are in pixels per *source* frame (independent of frame skipping)
but depend on resolution and frame rate. Classification uses ``speed_frac_per_s``
instead: image diagonals per second, which means the same thing in a 640-pixel YouTube
clip and a 4K recording.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .models import Track

REFERENCE_DIAGONAL = float(np.hypot(1920, 1080))  # used when the frame size is unknown

FEATURE_NAMES: tuple[str, ...] = (
    "duration",
    "detection_count",
    "avg_speed",
    "max_speed",
    "min_speed",
    "speed_consistency",
    "total_distance",
    "linearity",
    "direction_changes",
    "direction_change_rate",
    "avg_area",
    "max_area",
    "min_area",
    "size_consistency",
    "avg_aspect_ratio",
    "aspect_ratio_std",
    "avg_acceleration",
    "max_acceleration",
    "avg_brightness",
    "brightness_std",
    "max_brightness",
    "brightness_consistency",
    "brightness_variation_coeff",
    "blinking_score",
    "speed_frac_per_s",
    "flash_count",
    "flash_rate_hz",
    "flash_regularity",
    "flash_score",
    "cloudiness",
    "path_deviation",
    "path_deviation_max",
    # Scale-free versions, used by the learned classifier (same meaning at any
    # resolution, frame rate or camera gain):
    "detection_rate",
    "speed_cv",
    "max_speed_ratio",
    "path_length_frac",
    "accel_rel",
    "snr_median",
    "snr_max",
    "snr_cv",
    # Is it one object moving, or noise linked up by the tracker?
    "motion_residual",
    "sudden_jump",
)


JUMP_PACE = 3.0  # a step this many times faster than the track's average is a jump


def motion_residual(frames: np.ndarray, pos: np.ndarray) -> float:
    """Median distance (px) of each inner detection from where its two neighbours put it,
    interpolating in time. A real object moves smoothly, so this is its centroid noise:
    0.2-0.5 px for 23 real satellite passes in compressed 1080p video. Noise the tracker
    picked up lands anywhere in its search window (8-21 px for noise taken for meteors)."""
    if len(pos) < 3:
        return 0.0
    w = (frames[1:-1] - frames[:-2]) / (frames[2:] - frames[:-2])
    predicted = pos[:-2] + w[:, None] * (pos[2:] - pos[:-2])
    return float(np.median(np.hypot(*(pos[1:-1] - predicted).T)))


def sudden_jump(frames: np.ndarray, step_len: np.ndarray) -> float:
    """Largest share of the path covered by one step taken much faster than the track's
    average pace. A track that sits at one spot, jumps, then sits at another is two
    specks (twinkling stars, noise) joined up, not an object; a real object that fades
    for a while and reappears further on moves at its usual pace across the gap."""
    total, span = float(step_len.sum()), float(frames[-1] - frames[0])
    if len(step_len) < 2 or total <= 0 or span <= 0:
        return 0.0
    share = step_len / total
    fast = share / (np.diff(frames) / span) > JUMP_PACE
    return float(share[fast].max()) if fast.any() else 0.0


def path_deviation(pos: np.ndarray, percentile: float = 80.0) -> float:
    """Sideways distance from the best straight line, relative to path length.

    Meteors travel in straight lines (well under 1%); insects, bats and birds close to the
    camera curve and wiggle. The 80th percentile is used rather than the maximum: an
    aircraft strobe briefly shifts the measured position (spikes), whereas a curving or
    wiggling path deviates over much of its length.
    """
    if len(pos) < 3:
        return 0.0
    centred = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vt[0]
    span = float(along.max() - along.min())
    if span < 1.0:
        return 0.0
    return float(np.percentile(np.abs(centred @ vt[1]), percentile) / span)


def _consistency(std: float, mean: float = 1.0) -> float:
    """Map a spread to (0, 1]: 1 = perfectly constant."""
    return 1.0 / (1.0 + std / max(mean, 1.0))


def flash_features(track: Track) -> dict[str, float]:
    """Detect aircraft strobes: short, repeated brightness spikes at a steady rhythm.

    A strobe is a sample whose contrast jumps well above the object's own running
    baseline. Planes flash ~0.5-2 times per second with regular spacing; satellites shine
    steadily (a rare single "flare" is not a rhythm); noise produces irregular spikes.
    """
    zero = {"flash_count": 0.0, "flash_rate_hz": 0.0, "flash_regularity": 0.0, "flash_score": 0.0}
    dets = track.detections
    duration = track.duration
    if len(dets) < 8 or duration < 1.5:
        return zero
    c = np.array([d.contrast for d in dets], dtype=float)
    t = np.array([(d.timestamp - dets[0].timestamp).total_seconds() for d in dets])
    per_s = len(dets) / duration
    window = max(5, int(per_s) | 1)  # about one second of samples, odd
    base = median_filter(c, size=window, mode="nearest")
    resid = c - base
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
    spike = (c > 1.4 * np.maximum(base, 1.0)) & (resid > max(3.0 * noise, 2.0))
    if spike.mean() > 0.4:  # "spikes" most of the time: variability, not strobes
        return zero
    # Group spiking samples into flashes; flashes < 0.3 s apart are one (airliner double
    # flash), so only the rhythm between flash *groups* is measured.
    starts = t[np.nonzero(spike & ~np.concatenate([[False], spike[:-1]]))[0]]
    flashes = [float(s) for i, s in enumerate(starts) if i == 0 or s - starts[i - 1] > 0.3]
    n = len(flashes)
    if n < 4:
        return {**zero, "flash_count": float(n)}
    intervals = np.diff(flashes)
    # The beat is the typical interval; faint footage misses some flashes, so a gap of
    # ~2 or 3 beats is still on-rhythm. Regularity = how well intervals fit whole beats.
    beat = float(np.median(intervals))
    beats = np.maximum(np.round(intervals / beat), 1.0)
    jitter = np.abs(intervals / beats - beat) / beat
    regularity = float(np.clip(1.0 - 2.5 * np.median(jitter), 0.0, 1.0))
    rate = 1.0 / beat
    # Anti-collision lights flash 40-100 times per minute; allow some margin.
    in_band = 0.5 <= rate <= 2.5
    score = min(1.0, n / 6.0) * regularity**2 if in_band and regularity >= 0.5 else 0.0
    return {
        "flash_count": float(n),
        "flash_rate_hz": rate,
        "flash_regularity": regularity,
        "flash_score": score,
    }


def extract_features(
    track: Track,
    direction_change_deg: float = 30.0,
    blink_change_threshold: float = 0.15,
    frame_diagonal: float | None = None,
    frame_skip: int = 1,
) -> dict[str, float]:
    dets = track.detections
    n = len(dets)
    pos = track.positions()
    frames = track.frames()
    areas = np.array([d.area for d in dets])
    aspects = np.array([d.aspect_ratio for d in dets])
    bright = np.array([d.mean_brightness for d in dets])
    peak = float(max(d.max_brightness for d in dets))

    steps = np.diff(pos, axis=0)
    step_len = np.hypot(steps[:, 0], steps[:, 1]) if n > 1 else np.zeros(0)
    dt = np.maximum(np.diff(frames), 1.0) if n > 1 else np.zeros(0)
    speeds = step_len / dt if n > 1 else np.zeros(0)

    total_distance = float(step_len.sum())
    straight = float(np.hypot(*(pos[-1] - pos[0]))) if n > 1 else 0.0
    linearity = straight / max(total_distance, 1.0) if n > 1 else 0.0

    # Count heading changes larger than the threshold.
    direction_changes = 0
    if n >= 3:
        a, b = steps[:-1], steps[1:]
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        valid = norms > 0
        cos = np.clip((a * b).sum(axis=1)[valid] / norms[valid], -1.0, 1.0)
        direction_changes = int((np.arccos(cos) > np.radians(direction_change_deg)).sum())

    speed_std = float(speeds.std()) if speeds.size > 1 else 0.0
    accel = np.abs(np.diff(speeds)) if speeds.size > 1 else np.zeros(0)

    avg_brightness = float(bright.mean())
    brightness_std = float(bright.std()) if n > 1 else 0.0
    variation = brightness_std / max(avg_brightness, 1.0)
    blink_pattern = 0.0
    if n >= 3 and avg_brightness > 1.0:
        avg_diff = float(np.abs(np.diff(bright)).mean())
        if avg_diff > avg_brightness * blink_change_threshold:
            blink_pattern = min(avg_diff / avg_brightness, 1.0)

    avg_area = float(areas.mean())
    avg_speed = float(speeds.mean()) if speeds.size else 0.0
    diag = frame_diagonal or REFERENCE_DIAGONAL
    snr = np.array([d.snr for d in dets])
    processed = (frames[-1] - frames[0]) / frame_skip + 1  # frames the detector looked at
    return {
        "detection_rate": min(1.0, n / processed),
        "speed_cv": float(speeds.std() / avg_speed) if avg_speed > 0 else 0.0,
        "max_speed_ratio": float(speeds.max() / avg_speed) if avg_speed > 0 else 0.0,
        "path_length_frac": total_distance / diag,
        "accel_rel": float(accel.mean() / avg_speed) if accel.size and avg_speed > 0 else 0.0,
        "snr_median": float(np.median(snr)),
        "snr_max": float(snr.max()),
        "snr_cv": float(snr.std() / snr.mean()) if snr.mean() > 0 else 0.0,
        "motion_residual": motion_residual(frames, pos),
        "sudden_jump": sudden_jump(frames, step_len),
        **flash_features(track),
        "speed_frac_per_s": avg_speed * track.fps / diag,
        # ~1 on clear sky; well above 3 when the "object" is moving cloud texture.
        "cloudiness": float(np.median([d.texture for d in dets])),
        "path_deviation": path_deviation(pos),
        "path_deviation_max": path_deviation(pos, 100.0),
        "duration": track.duration,
        "detection_count": float(n),
        "avg_speed": avg_speed,
        "max_speed": float(speeds.max()) if speeds.size else 0.0,
        "min_speed": float(speeds.min()) if speeds.size else 0.0,
        "speed_consistency": 1.0 / (1.0 + speed_std) if speeds.size else 0.0,
        "total_distance": total_distance,
        "linearity": linearity,
        "direction_changes": float(direction_changes),
        "direction_change_rate": direction_changes / n,
        "avg_area": avg_area,
        "max_area": float(areas.max()),
        "min_area": float(areas.min()),
        "size_consistency": _consistency(float(areas.std()), avg_area),
        "avg_aspect_ratio": float(aspects.mean()),
        "aspect_ratio_std": float(aspects.std()),
        "avg_acceleration": float(accel.mean()) if accel.size else 0.0,
        "max_acceleration": float(accel.max()) if accel.size else 0.0,
        "avg_brightness": avg_brightness,
        "brightness_std": brightness_std,
        "max_brightness": peak,
        "brightness_consistency": _consistency(brightness_std, avg_brightness),
        "brightness_variation_coeff": variation,
        "blinking_score": max(variation, blink_pattern),
    }


def velocity(track: Track) -> tuple[float, float]:
    """Mean velocity (vx, vy) in px per source frame, from first to last detection."""
    if len(track.detections) < 2:
        return 0.0, 0.0
    first, last = track.detections[0], track.detections[-1]
    span = max(last.frame_index - first.frame_index, 1)
    return (last.x - first.x) / span, (last.y - first.y) / span
