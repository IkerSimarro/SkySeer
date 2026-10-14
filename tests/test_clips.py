import json

import cv2
import numpy as np
from typer.testing import CliRunner

from skyseer.cli import app
from skyseer.clips import ClipRecorder, FrameBuffer, pick_frames
from skyseer.config import load_config
from skyseer.pipeline import run
from skyseer.sources import FileSource


def test_frame_buffer_keeps_recent_frames_and_thins_old_ones():
    buf = FrameBuffer(recent=10, max_age=200)
    img = np.zeros((2, 2), np.uint8)
    for i in range(1001):
        buf.push(i, img)
    kept = [i for i, _ in buf.window(0, 1000)]
    assert kept[-10:] == list(range(991, 1001))  # newest frames all present
    assert min(kept) >= 1000 - 200 - 8  # nothing much older than max_age
    assert len(buf) <= FrameBuffer.capacity(10, 200) + 8  # memory stays bounded
    for i in kept:  # older frames are thinned to powers of two
        age = 1000 - i
        if age >= 40:
            assert i % 4 == 0


def test_pick_frames_gives_an_even_time_lapse():
    frames = [(i, None) for i in range(0, 400) if i >= 200 or i % 4 == 0]
    picked, stride = pick_frames(frames, max_frames=1000)  # type: ignore[arg-type]
    assert stride == 4  # the sparse old part sets the pace for the whole clip
    picked, stride = pick_frames(frames, max_frames=30)  # type: ignore[arg-type]
    assert len(picked) <= 30 and stride >= 16


def test_every_event_gets_a_clip_and_images(synth_clip, tmp_path):
    path, truth = synth_clip
    cfg = load_config()
    recorder = ClipRecorder(tmp_path, cfg.clips, 30.0)
    events = list(run(FileSource(path), cfg, clips=recorder))
    reported = [e for e in events if e.label not in ("Junk", "Star")]
    assert len(reported) == len(truth.objects)
    assert recorder.written == len(reported)
    for e in reported:
        media = e.to_dict()["media"]
        cap = cv2.VideoCapture(str(tmp_path / media["clip"]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == cfg.clips.width
        cap.release()
        # pre-roll + event + post-roll (within the video), divided by the time-lapse factor
        first, last = max(e.track.first_frame - 30, 0), min(e.track.last_frame + 30, 479)
        expected = (last - first + 1) / media["speedup"]
        assert abs(n - expected) <= 3, (n, expected)
        assert not media["partial"]
        thumb = cv2.imread(str(tmp_path / media["thumbnail"]), cv2.IMREAD_GRAYSCALE)
        context = cv2.imread(str(tmp_path / media["context"]))
        assert thumb.shape == (360, 640) and context is not None
        # The trail image shows the object's path as a bright line on a dark sky.
        assert np.percentile(thumb, 99.5) > 4 * np.median(thumb)


def test_long_events_become_time_lapses(synth_clip, tmp_path):
    path, _ = synth_clip
    cfg = load_config(clips={"max_frames": 40})
    recorder = ClipRecorder(tmp_path, cfg.clips, 30.0)
    events = [e for e in run(FileSource(path), cfg, clips=recorder) if e.media]
    longest = max(events, key=lambda e: e.track.duration)
    assert longest.media["speedup"] >= 8
    cap = cv2.VideoCapture(str(tmp_path / longest.media["clip"]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 40
    cap.release()


def test_buffer_respects_its_memory_budget(synth_clip, tmp_path):
    path, _ = synth_clip
    cfg = load_config(clips={"buffer_mb": 5})
    recorder = ClipRecorder(tmp_path, cfg.clips, 30.0)
    events = [e for e in run(FileSource(path), cfg, clips=recorder) if e.media]
    assert recorder._scale is not None and recorder._scale < 0.5
    assert events and all((tmp_path / e.media["clip"]).exists() for e in events)


def test_cli_analyze_writes_clips(synth_clip, tmp_path):
    path, _ = synth_clip
    result = CliRunner().invoke(app, ["analyze", str(path), "-o", str(tmp_path), "--no-video"])
    assert result.exit_code == 0, result.output
    events = json.loads((tmp_path / "events.json").read_text())
    assert all((tmp_path / e["media"]["clip"]).exists() for e in events)
