"""The Raspberry Pi camera source, against a stand-in for picamera2 (no Pi needed)."""

import sys
import time
import types
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from skyseer.pi_camera import PiCameraSettings, PiCameraSource, frame_interval_check


def _clock_ns() -> int:  # the clock libcamera's SensorTimestamp uses
    if sys.platform == "linux":
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return time.monotonic_ns()


class FakeRequest:
    def __init__(self, w, h, ts, exposure):
        self.w, self.h, self.ts, self.exposure = w, h, ts, exposure
        self.released = False

    def make_array(self, name):
        assert name == "main"
        yuv = np.zeros((self.h * 3 // 2, self.w), np.uint8)  # Y plane, then U and V
        yuv[: self.h] = 40
        yuv[self.h :] = 128
        return yuv

    def get_metadata(self):
        return {"SensorTimestamp": self.ts, "ExposureTime": self.exposure}

    def release(self):
        self.released = True


class FakePicamera2:
    instances: list = []

    def __init__(self, camera_num=0):
        self.camera_num = camera_num
        self.started = self.closed = False
        self.next_ts = _clock_ns()
        FakePicamera2.instances.append(self)

    def create_video_configuration(self, **kwargs):
        return kwargs

    def configure(self, config):
        self.config = config

    def start(self):
        self.started = True

    def capture_request(self):
        controls = self.config["controls"]
        w, h = self.config["main"]["size"]
        request = FakeRequest(w, h, self.next_ts, controls["ExposureTime"])
        self.next_ts += controls["FrameDurationLimits"][0] * 1000  # µs -> ns
        return request

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_picamera2(monkeypatch):
    module = types.ModuleType("picamera2")
    module.Picamera2 = FakePicamera2
    monkeypatch.setitem(sys.modules, "picamera2", module)
    FakePicamera2.instances.clear()
    return FakePicamera2


def test_night_settings_switch_off_everything_automatic(fake_picamera2):
    source = PiCameraSource(PiCameraSettings(fps=10, gain=12.0))
    cam = fake_picamera2.instances[0]
    controls = cam.config["controls"]
    assert cam.config["main"] == {"size": (2028, 1520), "format": "YUV420"}
    assert controls["AeEnable"] is False and controls["AwbEnable"] is False
    assert controls["ExposureTime"] == 95_000  # 95% of the 100 ms frame time
    assert controls["FrameDurationLimits"] == (100_000, 100_000)
    assert controls["AnalogueGain"] == 12.0 and controls["NoiseReductionMode"] == 0
    assert cam.started
    source.close()
    assert cam.closed and not cam.started


def test_frames_are_brightness_only_with_sensor_timestamps(fake_picamera2):
    source = PiCameraSource(PiCameraSettings(width=1332, height=990, fps=10))
    frames = []
    for frame in source:
        frames.append(frame)
        if len(frames) == 20:
            break
    assert frames[0].image.shape == (990, 1332) and frames[0].image.dtype == np.uint8
    assert int(frames[0].image.mean()) == 40  # the Y plane only, not the colour planes
    times = [f.timestamp for f in frames]
    assert frame_interval_check(times) == timedelta(milliseconds=100)
    # Mid-exposure time of the first frame: about now (the fake started its clock now).
    assert abs((times[0] - datetime.now(UTC)).total_seconds()) < 1.0
    assert [f.index for f in frames] == list(range(20))
    source.close()
