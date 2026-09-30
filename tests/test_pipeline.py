import json

import cv2
from typer.testing import CliRunner

from skyseer.cli import app
from skyseer.config import load_config
from skyseer.evaluate import evaluate
from skyseer.pipeline import run
from skyseer.render import render_annotated_video
from skyseer.sources import FileSource


def test_end_to_end_on_synthetic_scene(synth_clip):
    path, truth = synth_clip
    events = list(run(FileSource(path), load_config()))
    result = evaluate(events, truth)
    assert result.recall == 1.0, result.confusion
    assert result.accuracy >= 0.8, result.confusion
    assert result.false_positives == 0


def test_frame_skip_still_detects_everything(synth_clip):
    path, truth = synth_clip
    events = list(run(FileSource(path), load_config(pipeline={"frame_skip": 2})))
    assert evaluate(events, truth).recall == 1.0


def test_event_timestamps_follow_start_time(synth_clip):
    from datetime import UTC, datetime

    path, _ = synth_clip
    start = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)
    events = list(run(FileSource(path, start_time=start), load_config()))
    for e in events:
        expected = e.track.first_frame / 30.0
        assert abs((e.track.start_time - start).total_seconds() - expected) < 1e-6


def test_render_writes_video(synth_clip, tmp_path):
    path, _ = synth_clip
    events = list(run(FileSource(path), load_config()))
    out = render_annotated_video(path, events, tmp_path / "out.mp4", speedup=10)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 0 < n < 16 * 30  # fast-forwarded, but event frames kept


def test_cli_analyze(synth_clip, tmp_path):
    path, truth = synth_clip
    result = CliRunner().invoke(app, ["analyze", str(path), "-o", str(tmp_path), "--no-video"])
    assert result.exit_code == 0, result.output
    events = json.loads((tmp_path / "events.json").read_text())
    assert len(events) == len(truth.objects)
    assert (tmp_path / "events.csv").exists()
