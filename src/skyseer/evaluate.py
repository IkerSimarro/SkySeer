"""Score pipeline output against synthetic ground truth."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

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
    result = EvalResult(len(truth.objects), 0, 0, 0)
    matched_events: set[int] = set()
    for obj in truth.objects:
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
