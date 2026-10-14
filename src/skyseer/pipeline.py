"""The streaming pipeline: frames in, classified events out."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from .alerts import Alerter
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
    from .clips import ClipRecorder

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
        self._diagonal = float(np.hypot(*frame_size))
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
        # Camera movement / scene changes: frame ranges where the whole picture changed.
        self.disturbances: list[list[int]] = []
        self.discarded_tracks = 0
        self._pad = round(config.pipeline.disturbance_padding_s * fps)
        self._pending: list[Track] = []  # finished tracks waiting out the padding
        self._first_index: int | None = None
        self._settle_until = -1
        self._shower_associator: Any = None
        self.alerter = Alerter(config.alerts, config.photometry)

    def process(self, frame: Frame) -> list[Event]:
        if self.autocal is not None:
            self.autocal.feed(frame)
            self.calibration = self.autocal.calibration
        if frame.index % self.config.pipeline.frame_skip:
            return []
        if self._first_index is None:
            self._first_index = frame.index
        detections = self.detector.detect(frame)
        if self._is_disturbed(frame.index):
            detections = []
        self.last_detections = detections
        self._pending.extend(self.tracker.update(detections, frame.index))
        return self._release(frame.index)

    def finish(self) -> list[Event]:
        """Flush tracks still open when the stream ends."""
        self._pending.extend(self.tracker.flush())
        return self._release(None)

    # ----------------------------------------------------------- disturbances
    def _is_disturbed(self, index: int) -> bool:
        """Detect frames where the camera moved or the scene changed, and recover."""
        if index < self._settle_until:
            # A freshly reset background model flags everything for its first frames;
            # ignore detections while it settles, without counting a new disturbance.
            return True
        warming_up = index - (self._first_index or 0) < self.fps * 0.5  # background forming
        if warming_up or self.detector.changed_fraction <= (
            self.config.pipeline.disturbance_fraction
        ):
            return False
        gap = 3 * self.config.pipeline.frame_skip
        if self.disturbances and index - self.disturbances[-1][1] <= gap:
            self.disturbances[-1][1] = index
        else:
            self.disturbances.append([index, index])
            log.info("Camera moved or scene changed at frame %d; ignoring it", index)
            self._camera_moved()
        self.detector.reset()  # learn the new view quickly instead of over ~15 s
        self._settle_until = index + max(2, round(0.3 * self.fps))
        return True

    def _camera_moved(self) -> None:
        if self.autocal is not None:
            self.autocal.camera_moved()
            self.calibration = None
        elif self.calibration is not None:
            log.warning(
                "The camera view changed: sky positions are no longer reliable and "
                "are dropped from here on"
            )
            self.calibration = None

    def _overlaps_disturbance(self, track: Track) -> bool:
        return any(
            track.first_frame <= end + self._pad and track.last_frame >= start - self._pad
            for start, end in self.disturbances
        )

    def _release(self, frame_index: int | None) -> list[Event]:
        """Report finished tracks once no disturbance can still follow within the padding."""
        if frame_index is None:
            ready, self._pending = self._pending, []
            frame_index = max((t.last_frame for t in ready), default=0)
        else:
            ready = [t for t in self._pending if frame_index - t.last_frame > self._pad]
            self._pending = [t for t in self._pending if frame_index - t.last_frame <= self._pad]
        clean = [t for t in ready if not self._overlaps_disturbance(t)]
        self.discarded_tracks += len(ready) - len(clean)
        return self._handle(clean, frame_index)

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
        features = extract_features(
            track, rules.direction_change_deg, rules.blink_change_threshold, self._diagonal
        )
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
        self._science(event)
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

    def _science(self, event: Event) -> None:
        """Photometry, shower association, light-curve periodicity and fireball alerts."""
        from .astro.lightcurve import find_period
        from .astro.photometry import event_photometry

        dets = event.track.detections
        cal = self.calibration
        if cal is not None and cal.zero_point is not None:
            event.photometry = event_photometry(
                [d.flux for d in dets], [d.saturated for d in dets], cal.zero_point
            )
        if (
            event.label is ObjectClass.METEOR
            and event.sky is not None
            and self.config.showers.enabled
        ):
            event.shower_checked = True
            event.shower = self._associator(cal.observer if cal else None).associate(event.sky)
        if event.label in (ObjectClass.SATELLITE, ObjectClass.PLANE) and event.track.duration >= 6:
            t = np.array([(d.timestamp - dets[0].timestamp).total_seconds() for d in dets])
            if event.photometry is not None:
                values = np.array([np.nan if m is None else m for m in event.photometry.magnitudes])
            else:  # relative brightness in magnitude units
                values = -2.5 * np.log10(np.maximum([d.contrast for d in dets], 1.0))
            event.periodicity = find_period(t, values)
        self.alerter.check(event)

    def _associator(self, observer: Any) -> Any:
        from .astro.showers import ShowerAssociator

        if self._shower_associator is None or self._shower_associator.observer != observer:
            self._shower_associator = ShowerAssociator(self.config.showers, observer)
        return self._shower_associator


def run(
    source: FrameSource,
    config: Config,
    progress: ProgressCallback | None = None,
    calibration: Calibration | None = None,
    autocal: AutoCalibrator | None = None,
    identifier: SatelliteIdentifier | None = None,
    stats: dict[str, Any] | None = None,
    clips: ClipRecorder | None = None,
) -> Iterator[Event]:
    """Run the pipeline over a whole source, yielding events as they complete.

    If ``stats`` is given it is filled at the end with the disturbances that were ignored
    (camera moved / scene changed) and how many tracks they invalidated. If ``clips`` is
    given, every frame is buffered and each event gets a clip and images (``Event.media``).
    """
    pipeline = Pipeline(
        config, source.fps, (source.width, source.height), calibration, autocal, identifier
    )
    try:
        for frame in source:
            if clips is not None:
                clips.feed(frame)
            for event in pipeline.process(frame):
                if clips is not None:
                    clips.save(event)
                yield event
            if progress and frame.index % 100 == 0:
                progress(frame.index)
        for event in pipeline.finish():
            if clips is not None:
                clips.save(event)
            yield event
    finally:
        source.close()
        if clips is not None:
            clips.close()
        if stats is not None:
            stats["disturbances"] = [
                (s / source.fps, e / source.fps) for s, e in pipeline.disturbances
            ]
            stats["discarded_tracks"] = pipeline.discarded_tracks
