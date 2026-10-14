"""Score pipeline output against synthetic ground truth."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .models import Event, ObjectClass
from .synth import GroundTruth, TruthObject

_IGNORED = {ObjectClass.JUNK, ObjectClass.STAR}


@dataclass
class EvalResult:
    n_truth: int
    detected: int  # truth objects matched by at least one event
    correct: int  # ...and whose best-matching event has the right label
    false_positives: int  # reported (non-junk) events matching no truth object
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)  # (true, predicted)

    @property
    def recall(self) -> float:
        return self.detected / self.n_truth if self.n_truth else 1.0

    @property
    def accuracy(self) -> float:
        """Classification accuracy over detected objects."""
        return self.correct / self.detected if self.detected else 0.0


def _match_score(event: Event, obj: TruthObject, tolerance: float) -> float:
    """Fraction of the event's detections lying within ``tolerance`` px of the truth path."""
    truth = {f: (x, y) for f, x, y in obj.points}
    hits = 0
    for d in event.track.detections:
        p = truth.get(d.frame_index)
        if p is not None and np.hypot(d.x - p[0], d.y - p[1]) <= tolerance:
            hits += 1
    return hits / len(event.track.detections)


def evaluate(
    events: list[Event], truth: GroundTruth, tolerance: float = 15.0, min_match: float = 0.5
) -> EvalResult:
    """Recall and accuracy over the real objects; any other event not labelled Junk or
    Star is a false positive (including distractors such as insects that the truth file
    lists as Junk)."""
    real = [o for o in truth.objects if o.label != str(ObjectClass.JUNK)]
    result = EvalResult(len(real), 0, 0, 0)
    matched_events: set[int] = set()
    for obj in real:
        scored = [(_match_score(e, obj, tolerance), e) for e in events]
        scored = [(s, e) for s, e in scored if s >= min_match]
        if not scored:
            result.confusion[(obj.label, "Missed")] += 1
            continue
        matched_events.update(e.id for _, e in scored)
        # An object split into several tracks counts once, judged by its longest piece.
        best = max(scored, key=lambda se: len(se[1].track.detections))[1]
        result.detected += 1
        result.correct += int(str(best.label) == obj.label)
        result.confusion[(obj.label, str(best.label))] += 1
    result.false_positives = sum(
        1 for e in events if e.id not in matched_events and e.label not in _IGNORED
    )
    return result


@dataclass
class BenchmarkResult:
    result: EvalResult
    frames: int
    seconds: float  # processing time

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict[str, float | int]:
        r = self.result
        return {
            "objects": r.n_truth,
            "recall": round(r.recall, 4),
            "accuracy": round(r.accuracy, 4),
            "false_positives": r.false_positives,
            "frames_per_second": round(self.fps, 1),
        }


def run_benchmark(
    config: Config,
    scenes: int = 10,
    seconds: float = 20.0,
    cache: Path = Path("data/bench"),
    on_scene: Callable[[int], None] | None = None,
) -> BenchmarkResult:
    """Detection and classification accuracy on synthetic scenes with known truth.

    Scenes are generated once (seeds 1000+) and cached, so results are comparable
    between runs and code versions.
    """
    import time

    from .pipeline import run
    from .sources import FileSource
    from .synth import SceneSpec, generate

    total = EvalResult(0, 0, 0, 0)
    frames = 0
    elapsed = 0.0
    for seed in range(scenes):
        if on_scene is not None:
            on_scene(seed)
        path = cache / f"scene_{seed:03d}_{int(seconds)}s.mp4"
        if path.exists() and path.with_suffix(".json").exists():
            truth = GroundTruth.load(path.with_suffix(".json"))
        else:
            truth = generate(SceneSpec(seconds=seconds, seed=1000 + seed), path)
        source = FileSource(path)
        frames += source.frame_count
        t0 = time.perf_counter()
        events = list(run(source, config))
        elapsed += time.perf_counter() - t0
        r = evaluate(events, truth)
        total.n_truth += r.n_truth
        total.detected += r.detected
        total.correct += r.correct
        total.false_positives += r.false_positives
        total.confusion.update(r.confusion)
    return BenchmarkResult(total, frames, elapsed)
