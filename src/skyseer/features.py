"""Per-track kinematic and photometric features.

Speeds are in pixels per *source* frame so they don't depend on frame skipping.
"""

from __future__ import annotations

import numpy as np

from .models import Track

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
)


def _consistency(std: float, mean: float = 1.0) -> float:
    """Map a spread to (0, 1]: 1 = perfectly constant."""
    return 1.0 / (1.0 + std / max(mean, 1.0))


def extract_features(
    track: Track, direction_change_deg: float = 30.0, blink_change_threshold: float = 0.15
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
    return {
        "duration": track.duration,
        "detection_count": float(n),
        "avg_speed": float(speeds.mean()) if speeds.size else 0.0,
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
