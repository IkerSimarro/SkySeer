"""Multi-object tracking with constant-velocity Kalman filters and optimal assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .models import Detection, Track

_H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
_UNMATCHABLE = 1e9


class _KalmanCV:
    """2-D constant-velocity Kalman filter. Time is measured in source frames."""

    def __init__(self, x: float, y: float, q: float, r: float):
        self.state = np.array([x, y, 0.0, 0.0])
        self.cov = np.diag([r * r, r * r, 1e4, 1e4])  # velocity unknown at start
        self.q = q
        self.R = np.eye(2) * r * r

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Predicted (state, covariance) ``dt`` frames ahead, without committing it."""
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Discrete white-noise acceleration model.
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q
        return F @ self.state, F @ self.cov @ F.T + Q

    def update(self, z: np.ndarray, dt: float) -> None:
        x_pred, p_pred = self.predict(dt)
        innovation = z - _H @ x_pred
        S = _H @ p_pred @ _H.T + self.R
        K = p_pred @ _H.T @ np.linalg.inv(S)
        self.state = x_pred + K @ innovation
        self.cov = (np.eye(4) - K @ _H) @ p_pred


class _ActiveTrack:
    def __init__(self, track: Track, det: Detection, cfg: TrackerConfig):
        self.track = track
        self.kf = _KalmanCV(det.x, det.y, cfg.process_noise, cfg.measurement_noise)
        self.last_frame = det.frame_index
        track.detections.append(det)

    @property
    def hits(self) -> int:
        return len(self.track.detections)

    def predicted_position(self, frame_index: int) -> np.ndarray:
        state, _ = self.kf.predict(frame_index - self.last_frame)
        return state[:2]

    def add(self, det: Detection) -> None:
        dt = det.frame_index - self.last_frame
        if self.hits == 1:
            # Seed velocity from the first two points instead of waiting for the filter.
            first = self.track.detections[0]
            self.kf.state = np.array([det.x, det.y, (det.x - first.x) / dt, (det.y - first.y) / dt])
            r2 = self.kf.R[0, 0]
            self.kf.cov = np.diag([r2, r2, 2 * r2 / dt**2, 2 * r2 / dt**2])
        else:
            self.kf.update(np.array([det.x, det.y]), dt)
        self.last_frame = det.frame_index
        self.track.detections.append(det)


class Tracker:
    """Associates detections frame by frame and emits each track as soon as it ends.

    Because tracks are released when they are lost (not at end of video), the tracker
    works the same on a finite file and on an endless live camera stream.
    """

    def __init__(self, config: TrackerConfig, fps: float):
        self.config = config
        self.fps = fps
        self.max_missed_frames = max(1, round(config.max_missed_seconds * fps))
        self.tentative_missed_frames = max(1, round(config.tentative_missed_seconds * fps))
        self._active: list[_ActiveTrack] = []
        self._next_id = 1

    @property
    def active_tracks(self) -> list[Track]:
        return [a.track for a in self._active]

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        """Feed one processed frame's detections. Returns tracks that just finished."""
        self._associate(detections, frame_index)
        finished = [
            a
            for a in self._active
            if frame_index - a.last_frame
            > (self.max_missed_frames if a.hits >= 3 else self.tentative_missed_frames)
        ]
        if finished:
            self._active = [a for a in self._active if a not in finished]
        return [a.track for a in finished]

    def flush(self) -> list[Track]:
        """End every active track (call when the stream ends)."""
        tracks = [a.track for a in self._active]
        self._active = []
        return tracks

    def _gate(self, active: _ActiveTrack, frame_index: int) -> float:
        """Matching window around a track's predicted position (px).

        Before the velocity is known the window is wide. Afterwards it is tight right after
        a detection and widens only with the gap, by the speed error the track could have
        accumulated, so a slow object can't suddenly "jump" onto a nearby fast one and
        a faint satellite can still be picked up again after fading for a few seconds.
        """
        c = self.config
        if active.hits == 1:
            return c.new_track_distance
        dt = max(frame_index - active.last_frame, 1)
        speed = float(np.hypot(*active.kf.state[2:]))
        widen = (c.gate_velocity_floor + c.gate_velocity_tolerance * speed) * dt
        return float(min(c.max_distance, max(c.gate_floor_px, widen)))

    def _associate(self, detections: list[Detection], frame_index: int) -> None:
        unmatched = set(range(len(detections)))
        if self._active and detections:
            cost = np.full((len(self._active), len(detections)), _UNMATCHABLE)
            det_xy = np.array([(d.x, d.y) for d in detections])
            for i, active in enumerate(self._active):
                dist = np.linalg.norm(det_xy - active.predicted_position(frame_index), axis=1)
                cost[i] = np.where(dist <= self._gate(active, frame_index), dist, _UNMATCHABLE)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] < _UNMATCHABLE:
                    self._active[r].add(detections[c])
                    unmatched.discard(c)
        for c in sorted(unmatched):
            track = Track(id=self._next_id, fps=self.fps)
            self._next_id += 1
            self._active.append(_ActiveTrack(track, detections[c], self.config))
