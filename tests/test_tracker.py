from skyseer.config import TrackerConfig
from skyseer.track import Tracker

from .conftest import make_detection


def _run(tracker: Tracker, frames: dict[int, list[tuple[float, float]]], last: int):
    finished = []
    for f in range(last + 1):
        dets = [make_detection(f, x, y) for x, y in frames.get(f, [])]
        finished += tracker.update(dets, f)
    return finished + tracker.flush()


def test_crossing_objects_keep_identity():
    """Two objects crossing paths: a nearest-neighbour tracker swaps them, Kalman doesn't."""
    frames = {}
    for f in range(60):
        frames[f] = [(100 + 4 * f, 200.0), (340 - 4 * f, 203.0)]
    tracks = _run(Tracker(TrackerConfig(), fps=30), frames, 59)
    assert len(tracks) == 2
    for t in tracks:
        xs = [d.x for d in t.detections]
        steps = {round(b - a) for a, b in zip(xs, xs[1:], strict=False)}
        assert steps in ({4}, {-4}), f"track {t.id} switched objects"


def test_each_detection_assigned_to_one_track():
    frames = {f: [(100.0 + f, 100.0), (103.0 + f, 100.0)] for f in range(20)}
    tracks = _run(Tracker(TrackerConfig(), fps=30), frames, 19)
    ids = [id(d) for t in tracks for d in t.detections]
    assert len(ids) == len(set(ids)) == 40


def test_track_emitted_when_lost_not_at_end():
    tracker = Tracker(TrackerConfig(max_missed_seconds=0.5), fps=30)
    emitted_at = None
    for f in range(100):
        dets = [make_detection(f, 50 + 2 * f, 50)] if f < 30 else []
        if tracker.update(dets, f):
            emitted_at = f
            break
    assert emitted_at is not None and 30 < emitted_at <= 30 + 16


def test_fast_object_tracked_with_frame_skip():
    """A meteor moving 25 px/frame, processed every 3rd frame (75 px jumps)."""
    frames = {f: [(20 + 25.0 * f, 100.0)] for f in range(0, 24, 3)}
    tracks = _run(Tracker(TrackerConfig(), fps=30), frames, 24)
    assert len(tracks) == 1 and len(tracks[0].detections) == 8
