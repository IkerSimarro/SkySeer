"""A stopped analysis carries on from its last save and ends with the same results."""

import json

import pytest

from skyseer import analysis
from skyseer.analysis import CHECKPOINT, PARTIAL, analyze_video, can_resume
from skyseer.config import load_config
from skyseer.evaluate import evaluate
from skyseer.synth import SceneSpec, generate

from .conftest import T0


@pytest.fixture(scope="module")
def spread_clip(tmp_path_factory):
    """24 s: two meteors, then (after the stop at 13 s) a plane and four more meteors."""
    path = tmp_path_factory.mktemp("spread") / "scene.mp4"
    spec = SceneSpec(width=640, height=360, seconds=24, seed=7, n_satellites=0, n_meteors=6)
    return path, generate(spec, path)


class Stop(Exception):
    """Stands in for the server closing mid-analysis."""


def _stop_at(frame: int):
    def progress(i: int, total: int) -> None:
        if i >= frame:
            raise Stop

    return progress


def _summary(out_dir) -> list[tuple[int, str, int, int]]:
    events = json.loads((out_dir / "events.json").read_text())
    return [(e["id"], e["label"], e["first_frame"], e["last_frame"]) for e in events]


@pytest.mark.parametrize("warmup_s", [90.0, 3.0])  # re-read from the start, or from mid-video
def test_resumed_analysis_ends_like_an_uninterrupted_one(
    spread_clip, tmp_path, monkeypatch, warmup_s
):
    path, truth = spread_clip
    monkeypatch.setattr(analysis, "RESUME_WARMUP_S", warmup_s)
    cfg = load_config()
    straight = analyze_video(path, cfg, tmp_path / "a", T0, sky=False, checkpoint_s=2.0)

    out = tmp_path / "b"
    with pytest.raises(Stop):
        analyze_video(path, cfg, out, T0, sky=False, checkpoint_s=2.0, progress=_stop_at(400))
    assert can_resume(out)
    saved = json.loads((out / CHECKPOINT).read_text())
    assert 3.0 * 30 < saved["frame"] <= 400  # saved every 2 s: resumes mid-video
    done_before = [json.loads(line) for line in (out / PARTIAL).read_text().splitlines()]

    resumed = analyze_video(path, cfg, out, sky=False, resume=True, checkpoint_s=2.0)
    assert not can_resume(out) and not (out / PARTIAL).exists()
    assert len(resumed.earlier) == 2  # the two meteors before the stop
    assert len(resumed.earlier) + len(resumed.reported) == len(straight.reported)
    assert [e["id"] for e in resumed.earlier] == [
        e["id"] for e in done_before if e["first_frame"] < saved["frame"]
    ]

    events = _summary(out)
    ids = [i for i, *_ in events]
    assert len(set(ids)) == len(ids)  # nothing reported twice
    assert sorted((label, first) for _, label, first, _ in events) == sorted(
        (label, first) for _, label, first, _ in _summary(tmp_path / "a")
    )
    # Every event's clip exists, under its own number.
    for e in json.loads((out / "events.json").read_text()):
        assert (out / e["media"]["clip"]).is_file()
        assert e["media"]["clip"].endswith(f"{e['id']:04d}.mp4")
    assert evaluate(straight.events, truth).recall == 1.0


def test_a_finished_analysis_has_nothing_to_resume(synth_clip, tmp_path):
    path, _ = synth_clip
    analyze_video(path, load_config(), tmp_path, T0, sky=False, clips=False)
    assert not can_resume(tmp_path)
    with pytest.raises(ValueError, match="No stopped analysis"):
        analyze_video(path, load_config(), tmp_path, sky=False, resume=True)
