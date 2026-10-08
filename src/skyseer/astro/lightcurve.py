"""Periodic brightness changes ("light curves").

A tumbling dead satellite or rocket body glints each time a flat surface catches the Sun,
so its brightness repeats with the rotation period. Measuring that period from the ground
is a standard space-situational-awareness technique. Uses a Lomb-Scargle periodogram,
which handles the uneven sampling of a track with missed detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import lombscargle


@dataclass
class Periodicity:
    period_s: float
    amplitude: float  # peak-to-peak of the folded light curve (magnitudes if calibrated)
    false_alarm: float  # probability that noise alone produces such a peak
    cycles: float  # number of periods observed

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_s": round(self.period_s, 3),
            "amplitude": round(self.amplitude, 2),
            "false_alarm": float(f"{self.false_alarm:.2e}"),
            "cycles": round(self.cycles, 1),
        }


def find_period(
    t: np.ndarray,
    values: np.ndarray,
    min_period: float = 0.3,
    min_cycles: float = 3.0,
    max_false_alarm: float = 1e-3,
    min_amplitude: float = 0.3,
) -> Periodicity | None:
    """Most significant period in an unevenly sampled series, or None if not convincing."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    n = len(t)
    duration = float(t.max() - t.min()) if n else 0.0
    if n < 30 or duration < min_cycles * min_period:
        return None
    # Remove slow trends (changing distance and sun angle) with a wide running median.
    window = max(5, (n // 6) | 1)
    resid = y - median_filter(y, size=window, mode="nearest")
    resid -= resid.mean()
    if resid.std() == 0:
        return None
    max_period = duration / min_cycles
    freqs = np.linspace(1.0 / max_period, 1.0 / min_period, 4000)
    power = lombscargle(t, resid, 2 * np.pi * freqs, normalize=True)
    k = int(np.argmax(power))
    p = float(np.clip(power[k], 0.0, 0.999999))
    # Single-frequency false-alarm probability for the normalised periodogram, then a
    # correction for the number of independent frequencies searched.
    fap_single = (1.0 - p) ** ((n - 3) / 2.0)
    m = max(1.0, (freqs[-1] - freqs[0]) * duration)
    fap = float(1.0 - (1.0 - fap_single) ** m)
    period = 1.0 / float(freqs[k])
    phase = ((t - t.min()) / period) % 1.0
    bins = np.linspace(0, 1, 11)
    folded = [
        np.median(resid[(phase >= a) & (phase < b)])
        for a, b in zip(bins[:-1], bins[1:], strict=True)
        if ((phase >= a) & (phase < b)).any()
    ]
    amplitude = float(np.max(folded) - np.min(folded)) if folded else 0.0
    if fap > max_false_alarm or amplitude < min_amplitude:
        return None
    return Periodicity(period, amplitude, fap, duration / period)
