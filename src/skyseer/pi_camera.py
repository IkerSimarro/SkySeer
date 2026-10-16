"""Raspberry Pi camera source (picamera2), set up for the night sky.

Daytime camera defaults ruin night observing, so everything that adapts is switched off:

- **No auto-exposure or auto-gain.** At night they hunt, the picture brightness changes
  from frame to frame, and the background model sees "motion" everywhere. Exposure and
  gain are fixed instead.
- **Exposure = frame time.** At 10 fps each frame collects light for almost 1/10 s, so
  faint stars and satellites build up signal (a longer exposure would streak meteors
  more but miss fewer faint objects; 10 fps is a common meteor-camera choice).
- **No in-camera noise reduction or sharpening**: they smear and erase faint objects.
- **Fixed colour balance**, so the sky's colour doesn't drift through the night.
- **Only the brightness plane** (the Y of YUV420) is used: no colour conversion, which
  saves CPU on the Pi, and stars need nothing more.
- **Timestamps from the sensor**: each frame's capture time comes from the camera's own
  clock (the moment its exposure started) mapped to UTC, plus half the exposure, rather
  than when Python happened to read the frame. Satellite timing depends on it.

picamera2 is installed with the operating system on Raspberry Pi OS
(``sudo apt install python3-picamera2``); it isn't available from PyPI for other systems.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import Frame

log = logging.getLogger(__name__)


@dataclass
class PiCameraSettings:
    width: int = 2028  # HQ camera (IMX477), 2x2 binned: full field of view, less noise
    height: int = 1520
    fps: float = 10.0
    exposure_us: int | None = None  # default: 95% of the frame time
    gain: float = 8.0  # analogue gain; higher = brighter but noisier
    colour_gains: tuple[float, float] = (2.0, 1.8)  # red, blue (fixed white balance)
    camera_num: int = 0
    buffer_count: int = 6


def _monotonic_to_utc_offset() -> float:
    """Seconds to add to the camera's clock (Linux CLOCK_BOOTTIME, which libcamera's
    SensorTimestamp uses) to get Unix time."""
    if sys.platform == "linux":
        return time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)
    return time.time() - time.monotonic()  # elsewhere (tests): the monotonic clock


class PiCameraSource:
    """Frames from a Raspberry Pi camera, with night-sky settings (see the module notes)."""

    def __init__(self, settings: PiCameraSettings | None = None):
        from picamera2 import Picamera2  # Raspberry Pi OS only

        self.settings = s = settings or PiCameraSettings()
        self.fps = s.fps
        self.width, self.height = s.width, s.height
        frame_us = int(round(1e6 / s.fps))
        self.exposure_us = s.exposure_us or int(0.95 * frame_us)
        self._camera = Picamera2(camera_num=s.camera_num)
        config = self._camera.create_video_configuration(
            main={"size": (s.width, s.height), "format": "YUV420"},
            buffer_count=s.buffer_count,
            controls=self.controls(),
        )
        self._camera.configure(config)
        self._camera.start()
        self.start_time = datetime.now(UTC)
        log.info(
            "Pi camera %d: %dx%d at %.1f fps, exposure %.1f ms, gain %.1f",
            s.camera_num,
            s.width,
            s.height,
            s.fps,
            self.exposure_us / 1000,
            s.gain,
        )

    def controls(self) -> dict[str, Any]:
        s = self.settings
        frame_us = int(round(1e6 / s.fps))
        return {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": self.exposure_us,
            "AnalogueGain": s.gain,
            "FrameDurationLimits": (frame_us, frame_us),  # a steady frame rate
            "ColourGains": s.colour_gains,
            "NoiseReductionMode": 0,  # off
            "Sharpness": 0.0,
        }

    def __iter__(self) -> Iterator[Frame]:
        offset = _monotonic_to_utc_offset()
        index = 0
        while True:
            request = self._camera.capture_request()
            try:
                image = request.make_array("main")[: self.height, : self.width].copy()
                meta = request.get_metadata()
            finally:
                request.release()
            ts = meta.get("SensorTimestamp")  # nanoseconds, start of exposure
            exposure = meta.get("ExposureTime", self.exposure_us)
            if ts is None:
                when = datetime.now(UTC)
            else:
                unix = ts / 1e9 + offset + exposure / 2e6  # mid-exposure
                when = datetime.fromtimestamp(unix, UTC)
            if index % 600 == 0:  # re-anchor occasionally, in case the system clock was set
                offset = _monotonic_to_utc_offset()
            yield Frame(image, index, when)
            index += 1

    def close(self) -> None:
        try:
            self._camera.stop()
        finally:
            self._camera.close()


def frame_interval_check(times: list[datetime]) -> timedelta:
    """Median spacing of frame timestamps (for diagnostics and tests)."""
    gaps = sorted(b - a for a, b in zip(times, times[1:], strict=False))
    return gaps[len(gaps) // 2] if gaps else timedelta(0)
