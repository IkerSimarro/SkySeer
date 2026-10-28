"""The Raspberry Pi camera source, against a stand-in for picamera2 (no Pi needed), and
against the real camera library where it is installed (Raspberry Pi OS, and the CI job
that reproduces it)."""

import inspect
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
    delivers: tuple[int, int] | None = None  # a size libcamera adjusted the request to
    connected = [{"Model": "imx477", "Num": 0}]

    @staticmethod
    def global_camera_info():
        return FakePicamera2.connected

    def __init__(self, camera_num=0):
        self.camera_num = camera_num
        self.started = self.closed = False
        self.next_ts = _clock_ns()
        FakePicamera2.instances.append(self)

    def create_video_configuration(self, **kwargs):
        return kwargs

    def configure(self, config):
        if FakePicamera2.delivers is not None:
            config = {**config, "main": {**config["main"], "size": FakePicamera2.delivers}}
        self.config = config

    def camera_configuration(self):
        return self.config

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
    FakePicamera2.delivers = None
    FakePicamera2.connected = [{"Model": "imx477", "Num": 0}]
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


def test_frames_have_the_size_the_camera_delivers(fake_picamera2):
    fake_picamera2.delivers = (2016, 1520)  # e.g. rounded for alignment
    source = PiCameraSource(PiCameraSettings(width=2028, height=1520))
    assert (source.width, source.height) == (2016, 1520)
    frame = next(iter(source))
    assert frame.image.shape == (1520, 2016)
    source.close()


def test_no_camera_says_what_to_check(fake_picamera2):
    fake_picamera2.connected = []  # a loose ribbon cable
    with pytest.raises(RuntimeError, match="No camera 0 found .0 connected.: check the ribbon"):
        PiCameraSource()
    assert fake_picamera2.instances == []  # never opened


def test_the_real_camera_library_takes_these_settings():
    """Every control the driver sets exists in libcamera, and picamera2 has the calls it
    makes. Runs only where the real library is installed (no camera needed)."""
    picamera2 = pytest.importorskip("picamera2")
    from libcamera import controls as lc
    from picamera2.request import CompletedRequest

    source = PiCameraSource.__new__(PiCameraSource)  # its settings, without a camera
    source.settings, source.exposure_us = PiCameraSettings(), 95_000
    known = set(dir(lc)) | set(dir(lc.draft))
    assert [name for name in source.controls() if name not in known] == []
    noise = getattr(lc, "NoiseReductionModeEnum", None) or lc.draft.NoiseReductionModeEnum
    assert noise.Off == source.controls()["NoiseReductionMode"]

    camera = picamera2.Picamera2
    takes = inspect.signature(camera.create_video_configuration).parameters
    assert {"main", "buffer_count", "controls"} <= set(takes)
    names = ("global_camera_info", "configure", "camera_configuration", "start", "capture_request")
    for name in (*names, "stop", "close"):
        assert callable(getattr(camera, name)), name
    for name in ("make_array", "get_metadata", "release"):
        assert callable(getattr(CompletedRequest, name)), name
