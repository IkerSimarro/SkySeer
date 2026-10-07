"""The streaming pipeline: frames in, classified events out."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from .classify import RuleClassifier
from .config import Config
from .detect import MotionDetector
from .features import extract_features
from .models import Detection, Event, Frame, ObjectClass, Track
from .sources import FrameSource
from .starfield import StarFieldFilter
from .track import Tracker

if TYPE_CHECKING:
    from .astro.autocal import AutoCalibrator
    from .astro.calibrate import Calibration
    from .astro.satellites import SatelliteIdentifier

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int], None]


class Pipeline:
    """source → detect → track → filter → classify.

    Call :meth:`process` once per frame; it returns the events completed on that frame.
    Nothing waits for the end of the video, so the same object serves offline files and
    a live camera.
    """

    def __init__(
        self,
        config: Config,
        fps: float,
        frame_size: tuple[int, int],
        calibration: Calibration | None = None,
        autocal: AutoCalibrator | None = None,
        identifier: SatelliteIdentifier | None = None,
    ):
        self.config = config
        self.calibration = calibration
        self.autocal = autocal
        self.identifier = identifier
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
        if self.autocal is not None:
            self.autocal.feed(frame)
            if self.autocal.calibration is not None:
                self.calibration = self.autocal.calibration
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
        if label is ObjectClass.STAR:
            event.label_source = "starfield"
        if self.calibration is not None:
            from .astro.sky import sky_path

            event.sky = sky_path(track, self.calibration)
            if (
                self.identifier is not None
                and label is not ObjectClass.STAR
                and self.identifier.can_check(event.sky)
            ):
                event.identification_checked = True
                match = self.identifier.identify(event.sky)
                if match is not None:
                    event.identification = match
                    if match.confidence >= 0.5:
                        # A confident orbit match outranks the motion rules.
                        event.label, event.label_source = ObjectClass.SATELLITE, "orbit"
                        event.confidence = match.confidence
                    log.info("Event %d identified as %s", event.id, match.short_label())
        log.info(
            "Event %d: %s (%.2f, %s) %.1fs, %d detections",
            event.id,
            event.label,
            event.confidence,
            event.label_source,
            track.duration,
            len(track.detections),
        )
        return event


def run(
    source: FrameSource,
    config: Config,
    progress: ProgressCallback | None = None,
    calibration: Calibration | None = None,
    autocal: AutoCalibrator | None = None,
    identifier: SatelliteIdentifier | None = None,
) -> Iterator[Event]:
    """Run the pipeline over a whole source, yielding events as they complete."""
    pipeline = Pipeline(
        config, source.fps, (source.width, source.height), calibration, autocal, identifier
    )
    try:
        for frame in source:
            yield from pipeline.process(frame)
            if progress and frame.index % 100 == 0:
                progress(frame.index)
        yield from pipeline.finish()
    finally:
        source.close()
