"""The streaming pipeline: frames in, classified events out."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator

from .classify import RuleClassifier
from .config import Config
from .detect import MotionDetector
from .features import extract_features
from .models import Detection, Event, Frame, ObjectClass, Track
from .sources import FrameSource
from .starfield import StarFieldFilter
from .track import Tracker

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int], None]


class Pipeline:
    """source → detect → track → filter → classify.

    Call :meth:`process` once per frame; it returns the events completed on that frame.
    Nothing waits for the end of the video, so the same object serves offline files and
    a live camera.
    """

    def __init__(self, config: Config, fps: float, frame_size: tuple[int, int]):
        self.config = config
        self.fps = fps
        self.detector = MotionDetector(config.detector, frame_size)
        self.tracker = Tracker(config.tracker, fps)
        self.star_filter = StarFieldFilter(config.star_filter)
        self.classifier = RuleClassifier(config.rules)
        self._recent: deque[Track] = deque()  # finished tracks kept as star-filter context
        self._context_frames = round(60 * fps)
        self._next_event_id = 1
        self.last_detections: list[Detection] = []

    def process(self, frame: Frame) -> list[Event]:
        if frame.index % self.config.pipeline.frame_skip:
            return []
        self.last_detections = self.detector.detect(frame)
        finished = self.tracker.update(self.last_detections, frame.index)
        return self._handle(finished, frame.index)

    def finish(self) -> list[Event]:
        """Flush tracks still open when the stream ends."""
        finished = self.tracker.flush()
        last = max((t.last_frame for t in finished), default=0)
        return self._handle(finished, last)

    def _handle(self, finished: list[Track], frame_index: int) -> list[Event]:
        while self._recent and frame_index - self._recent[0].last_frame > self._context_frames:
            self._recent.popleft()
        kept = [t for t in finished if self._passes_filter(t)]
        self._recent.extend(kept)
        context = list(self._recent) + self.tracker.active_tracks
        return [self._classify(t, context) for t in kept]

    def _passes_filter(self, track: Track) -> bool:
        f = self.config.filter
        if len(track.detections) < f.min_detections or track.duration < f.min_duration:
            return False
        return f.max_duration is None or track.duration <= f.max_duration

    def _classify(self, track: Track, context: list[Track]) -> Event:
        rules = self.config.rules
        features = extract_features(track, rules.direction_change_deg, rules.blink_change_threshold)
        if self.star_filter.is_star(track, context):
            label, confidence, scores = ObjectClass.STAR, 0.9, {}
        else:
            result = self.classifier.classify(features)
            label, confidence, scores = result.label, result.confidence, result.scores
        event = Event(self._next_event_id, track, label, confidence, features, scores)
        self._next_event_id += 1
        log.info(
            "Event %d: %s (%.2f) %.1fs, %d detections",
            event.id,
            label,
            confidence,
            track.duration,
            len(track.detections),
        )
        return event


def run(
    source: FrameSource, config: Config, progress: ProgressCallback | None = None
) -> Iterator[Event]:
    """Run the pipeline over a whole source, yielding events as they complete."""
    pipeline = Pipeline(config, source.fps, (source.width, source.height))
    try:
        for frame in source:
            yield from pipeline.process(frame)
            if progress and frame.index % 100 == 0:
                progress(frame.index)
        yield from pipeline.finish()
    finally:
        source.close()
