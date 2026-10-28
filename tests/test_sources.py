"""The PyAV video reader gives the same frames, numbered the same, as OpenCV's."""

import struct
from pathlib import Path

import av
import cv2
import numpy as np

from skyseer.sources import FileSource, VideoFile


def _opencv_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(image)
    cap.release()
    return frames


def test_reader_matches_opencv(synth_clip):
    path, _ = synth_clip
    expected = _opencv_frames(path)
    video = VideoFile(path)
    assert (video.fps, video.width, video.height) == (30.0, 640, 360)
    assert video.frame_count == len(expected) == 16 * 30
    got = list(video.frames())
    video.close()
    assert [i for i, _ in got] == list(range(len(expected)))
    for (_, image), ref in zip(got, expected, strict=True):
        assert image.shape == ref.shape
        assert np.abs(image.astype(int) - ref).mean() < 0.5


def test_seeking_gives_the_frames_reading_through_does(synth_clip):
    path, _ = synth_clip
    video = VideoFile(path)
    through = dict(video.frames(0, 260))
    part = list(video.frames(250, 260))
    video.close()
    assert [i for i, _ in part] == list(range(250, 260))
    for index, image in part:
        assert np.array_equal(image, through[index])


def test_greyscale_matches_converting_colour(synth_clip):
    path, _ = synth_clip
    video = VideoFile(path)
    colour = dict(video.frames(100, 110))
    grey = list(video.frames(100, 110, gray=True))
    video.close()
    for index, image in grey:
        assert image.ndim == 2
        ref = cv2.cvtColor(colour[index], cv2.COLOR_BGR2GRAY)
        assert np.abs(image.astype(int) - ref).mean() < 2  # YUV's own luma vs re-weighted BGR


def test_file_source_keeps_full_video_numbering(synth_clip):
    path, _ = synth_clip
    source = FileSource(path, from_s=2.0, to_s=3.0, gray=True)
    frames = list(source)
    source.close()
    assert [f.index for f in frames] == list(range(60, 90))
    assert frames[0].image.ndim == 2
    assert abs((frames[-1].timestamp - frames[0].timestamp).total_seconds() - 29 / 30) < 1e-5


def _rotated_video(path: Path) -> None:
    """A landscape video whose header says "show me turned 90°", as phones write."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        image = np.zeros((32, 64, 3), np.uint8)
        image[:, :16] = 255  # a white band on the left
        for _ in range(5):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    data = bytearray(path.read_bytes())
    at = data.find(b"tkhd")
    # Track header: version/flags, times and ids (v0), 8 reserved, layer/group/volume/reserved.
    matrix = at + 4 + 4 + (20 if data[at + 4] == 0 else 32) + 8 + 8
    data[matrix : matrix + 36] = struct.pack(">9i", 0, 0x10000, 0, -0x10000, 0, 0, 0, 0, 1 << 30)
    path.write_bytes(bytes(data))


def test_rotated_video_comes_out_upright(tmp_path):
    path = tmp_path / "phone.mp4"
    _rotated_video(path)
    video = VideoFile(path)
    _, image = next(video.frames())
    _, grey = next(video.frames(gray=True))
    video.close()
    assert (video.width, video.height) == (32, 64)
    assert image.shape == (64, 32, 3) and grey.shape == (64, 32)
    # A quarter turn clockwise: the band on the left of the stored picture is now on top.
    assert grey[:14].mean() > 240 and grey[18:].mean() < 15
    # OpenCV builds that read the rotation agree pixel for pixel (OpenCV 4.11's ARM wheel
    # leaves the picture sideways, so it isn't the reference).
    ref = _opencv_frames(path)[0]
    if ref.shape == image.shape:
        assert np.abs(image.astype(int) - ref).mean() < 2
