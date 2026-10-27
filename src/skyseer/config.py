"""Typed configuration. Defaults live here; YAML files override any subset of them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Section):
    sensitivity: int = Field(5, ge=1, le=10, description="1 = only bright blobs, 10 = faintest")
    process_scale: float = Field(
        1.0,
        gt=0,
        le=1,
        description="Downscale factor applied before detection (0.5 = half size: ~3x faster "
        "on HD video, but faint satellites can be missed on a bright sky)",
    )
    gpu: bool = Field(
        True,
        description="Run the background model on the graphics card (OpenCL) when there is "
        "one: the same detections, faster",
    )
    mog2_history: int = 500
    mog2_var_threshold: float = 35.0
    blur_kernel: int = Field(3, description="1 = no blur; larger blurs dim 1-2 px objects")
    morph_kernel: int = Field(
        1, description="Mask clean-up kernel; >1 erases objects smaller than the kernel"
    )
    group_px: int = Field(
        1,
        ge=0,
        description="Blob pieces up to 2x this many px apart count as one object (noise can "
        "break a faint object up); 0 = off",
    )
    max_area_fraction: float = Field(
        0.01,
        description="Blobs larger than this fraction of the frame are ignored (clouds, lights)",
    )

    min_area_px: float | None = Field(
        None, description="Override the minimum blob area (px) that sensitivity implies"
    )

    @property
    def min_area(self) -> float:
        """Minimum blob area in full-resolution pixels, derived from sensitivity."""
        if self.min_area_px is not None:
            return self.min_area_px
        # Faint satellites can be 1-2 px wide; noise blobs are rejected later by tracking.
        return max(1.0, 12.0 - self.sensitivity * 2.0)


class TrackerConfig(_Section):
    max_distance: float = Field(
        60.0, description="Gate (px) between a track's predicted position and a detection"
    )
    new_track_distance: float = Field(
        150.0, description="Gate (px) for the second detection, before velocity is known"
    )
    max_missed_seconds: float = Field(
        3.0,
        description="Seconds unseen before a track ends; faint satellites fade in and out, "
        "and the motion model bridges the gaps",
    )
    tentative_missed_seconds: float = Field(
        0.3, description="Unconfirmed tracks (1-2 detections, maybe noise) end sooner"
    )
    gate_floor_px: float = Field(8.0, description="Smallest matching window around a prediction")
    gate_velocity_tolerance: float = Field(
        0.5, description="Allowed relative speed error per frame of gap (widens the window)"
    )
    gate_velocity_floor: float = Field(0.3, description="Allowed speed error, px/frame")
    process_noise: float = 1.0
    measurement_noise: float = 2.0
    merge_area_ratio: float = Field(
        1.4,
        description="A blob inside two tracks' windows and this many times bigger than their "
        "usual size is the two objects overlapping: neither track uses it (0 = off)",
    )


class FilterConfig(_Section):
    min_duration: float = Field(0.3, ge=0, description="Tracks shorter than this are dropped (s)")
    max_duration: float | None = Field(None, description="Tracks longer than this are dropped (s)")
    min_detections: int = Field(3, ge=2)


class StarFilterConfig(_Section):
    enabled: bool = True
    # Star-field drift is a rigid motion, so real stars share almost identical velocities;
    # loose tolerances wrongly group unrelated objects that happen to move alike.
    min_group_size: int = 4
    velocity_tolerance: float = 0.1
    direction_tolerance_deg: float = 10.0
    min_temporal_overlap: float = 0.5
    max_spatial_distance: float = 500.0
    min_speed: float = 0.01


class ClassifierConfig(_Section):
    method: Literal["rules", "model"] = Field(
        "rules", description="rules = hand-written baseline; model = learned (skyseer train)"
    )
    model_path: str = Field(
        "data/models/classifier.joblib", description="Trained model used by method: model"
    )


class RulesConfig(_Section):
    """Thresholds for the heuristic baseline classifier.

    Speeds are in image diagonals per second, so the same rules work at any resolution and
    frame rate. Durations are in seconds.
    """

    meteor_reference_speed: float = Field(0.4, description="Diagonals/s considered 'fast'")
    meteor_max_duration: float = 3.0
    # (upper_bound_s, factor): only very short tracks are penalised. How long a satellite
    # stays in view depends on the lens (minutes for an all-sky camera), so long is fine.
    satellite_duration_bands: list[tuple[float, float]] = [(1.0, 0.3), (2.0, 0.6), (3.0, 0.85)]
    satellite_long_factor: float = 1.0
    plane_duration_bands: list[tuple[float, float]] = [(1.5, 0.5)]
    plane_long_factor: float = 1.0
    plane_base_score: float = Field(0.1, description="Plane score with no strobes seen")
    cloud_texture: float = Field(
        3.0, description="Surroundings this many times noisier than clear sky = cloud"
    )
    blink_change_threshold: float = 0.15
    direction_change_deg: float = 30.0
    junk_base_score: float = 0.1
    stationary_speed: float = Field(
        0.0015, description="Diagonals/s below which a track is noise or a star"
    )
    smooth_px: float = Field(
        2.0,
        gt=0,
        description="Detections this far (median px) from where their neighbours put them "
        "make a track doubtful; at twice this it is noise linked up",
    )
    plane_smooth_px: float = Field(3.0, gt=0, description="The same for aircraft (strobes)")
    jump_share: float = Field(
        0.5,
        gt=0,
        le=1,
        description="A track covering this share of its path in one sudden jump is two "
        "specks joined up (trusted fully below 60% of it)",
    )
    evidence_rate: float = Field(
        0.3, gt=0, description="A satellite or aircraft is seen in this share of its frames..."
    )
    evidence_count: float = Field(40.0, gt=0, description="...or at least this many times")


class ObserverConfig(_Section):
    """Where the camera is. Filled in by hand, or automatically from GPS on the station."""

    latitude: float | None = Field(None, ge=-90, le=90, description="Degrees, north positive")
    longitude: float | None = Field(None, ge=-180, le=180, description="Degrees, east positive")
    elevation_m: float = 0.0

    @property
    def known(self) -> bool:
        return self.latitude is not None and self.longitude is not None


class AstroConfig(_Section):
    fov_estimate: float | None = Field(
        None, description="Approximate horizontal field of view in degrees (speeds up solving)"
    )
    fov_tolerance: float = 10.0
    calibration_seconds: float = Field(2.0, description="Video stacked for a calibration")
    database_min_fov: float = 20.0
    database_max_fov: float = 70.0
    database_max_mag: float = 6.5
    data_dir: str = "data/astro"


class SatelliteConfig(_Section):
    """Identifying detections against the public satellite catalogue (needs the location)."""

    enabled: bool = True
    groups: list[str] = Field(
        default_factory=lambda: ["active", "visual"],
        description="CelesTrak groups: 'active' payloads plus 'visual' (bright rocket bodies)",
    )
    max_age_hours: float = Field(24.0, description="Re-download orbits older than this")
    extra_tle_files: list[str] = Field(
        default_factory=list, description="Extra TLE files, e.g. historical sets for old videos"
    )
    max_clock_offset_s: float = Field(
        3.0, description="How wrong the camera clock may be (GPS-timed stations: ~0.1 s)"
    )
    max_cross_track_deg: float = 0.3
    max_timing_spread_s: float = 0.5
    candidate_radius_deg: float = 2.0
    max_candidates: int = 40
    max_elements_age_days: float = Field(
        10.0, description="Skip identification if orbital elements are further from the date"
    )


class AircraftConfig(_Section):
    """Naming aircraft from their ADS-B broadcasts (needs the location)."""

    enabled: bool = True
    source: str = Field(
        "none",
        description="Live broadcasts: none, opensky (free online service; set "
        "SKYSEER_OPENSKY_CLIENT_ID/_SECRET for an account) or readsb (a local receiver)",
    )
    readsb_url: str = Field(
        "http://localhost/tar1090/data/aircraft.json",
        description="readsb/dump1090 aircraft.json (URL or file path)",
    )
    radius_km: float = Field(150.0, description="Ignore aircraft further from the camera")
    poll_s: float = Field(
        10.0, description="Seconds between requests (OpenSky: 10 without an account, 5 with)"
    )
    active_only: bool = Field(
        True, description="Ask only while something is crossing the view (saves OpenSky credits)"
    )
    max_requests_per_day: int = Field(
        380, description="Stop asking OpenSky for the day after this (anonymous limit: 400)"
    )
    check_delay_s: float = Field(
        15.0, description="Live: wait this long after an event for broadcasts that cover it"
    )
    max_cross_track_deg: float = 0.5
    min_arc_deg: float = Field(
        0.5,
        description="Paths shorter than this aren't matched: their direction can't be checked, "
        "so a speck near a distant aircraft would 'match' it",
    )
    max_clock_offset_s: float = 3.0
    max_timing_spread_s: float = 1.0
    candidate_radius_deg: float = 8.0
    max_gap_s: float = Field(30.0, description="Interpolate between broadcasts at most this apart")
    max_extrapolate_s: float = Field(
        15.0, description="Dead-reckon from speed and heading at most this far past a broadcast"
    )


class ShowerConfig(_Section):
    """Associating meteors with showers (IAU Meteor Data Center established showers)."""

    enabled: bool = True
    # Defaults chosen by Monte Carlo (2,000 simulated meteors with realistic radiant and
    # speed scatter): ~96% of shower meteors found, ~5% of sporadics matched by chance.
    max_radiant_offset_deg: float = Field(
        2.5, description="How close the backward path must pass the radiant (+ path error)"
    )
    direction_error_deg: float = Field(
        0.05, description="Typical position error at each end of a meteor path (deg)"
    )
    max_offset_cap_deg: float = 6.0
    min_arc_deg: float = Field(1.0, description="Shorter paths can't be traced back reliably")
    min_radiant_distance_deg: float = 2.0
    max_radiant_distance_deg: float = 90.0
    min_radiant_elevation_deg: float = -5.0
    min_speed_ratio: float = 0.6
    max_speed_ratio: float = 1.5
    min_members: int = Field(
        0, description="Ignore showers detected with fewer meteors than this (minor showers)"
    )


class PhotometryConfig(_Section):
    fireball_magnitude: float = Field(
        -4.0, description="Meteors this bright or brighter are fireballs (Venus is about -4)"
    )
    aperture_px: float = 4.0


class AlertConfig(_Section):
    """Where fireball alerts go. Nothing is sent anywhere unless configured here."""

    log_file: str | None = "data/alerts.jsonl"
    webhook_url: str | None = Field(
        None, description="POST a JSON message here (e.g. a home-automation webhook)"
    )
    ntfy_topic: str | None = Field(
        None, description="Push to phones via the free ntfy app (https://ntfy.sh/<topic>)"
    )


class StabiliseConfig(_Section):
    """Camera-shake correction measured from the stars (see stabilise.py)."""

    enabled: bool = True
    max_stars: int = Field(60, ge=3, description="Reference stars measured every frame")
    min_stars: int = Field(8, ge=3, description="Fewer measurable stars: no correction")
    min_star_snr: float = Field(6.0, gt=0, description="Star peak above the sky, in noise units")
    search_px: int = Field(
        15, ge=2, description="Coarse search radius: the most a star may move between frames"
    )
    window_px: int = Field(5, ge=2, description="Window for each star's sub-pixel centre")
    agree_px: float = Field(
        2.5, gt=0, description="Stars agreeing on the shift within this are still tracked"
    )
    keyframe_s: float = Field(10.0, gt=0, description="Pick fresh reference stars this often")
    trend_s: float = Field(
        3.0, gt=0, description="Motion slower than this is drift, left to the background model"
    )
    step_px: float = Field(
        3.0, gt=0, description="A sudden jump this big (beyond the shaking) is a knock: corrected"
    )
    confirm_s: float = Field(
        0.3, ge=0, description="Stars suddenly lost this long: the camera moved (glitches recover)"
    )
    min_shift_px: float = Field(0.25, ge=0, description="Smaller jitter is harmless: not corrected")
    max_shift_px: float = Field(
        15.0, gt=0, description="Bigger jumps are real camera moves, handled as disturbances"
    )


class ClipConfig(_Section):
    """Short video clip, trail image and context image written for every event."""

    enabled: bool = True
    pre_roll_s: float = Field(1.0, ge=0, description="Clip starts this long before the object")
    post_roll_s: float = Field(
        1.0, ge=0, description="...and ends this long after (up to the disturbance padding)"
    )
    max_frames: int = Field(300, ge=10, description="Longer events become time-lapses")
    recent_seconds: float = Field(
        8.0, gt=0, description="Frames this recent are all buffered; older ones are thinned"
    )
    buffer_seconds: float = Field(
        120.0, gt=0, description="Oldest frame buffered: longer events lose their start"
    )
    buffer_mb: float = Field(
        1024.0, gt=0, description="Memory budget: frames are stored smaller beyond it"
    )
    color: bool = Field(False, description="Keep colour (3x the memory of grey frames)")
    width: int = Field(640, ge=64, description="Width of clips and images in pixels")
    margin_px: float = Field(40.0, ge=0, description="Space around the path in the zoomed clip")
    min_crop_px: float = Field(200.0, ge=16, description="Smallest region a clip zooms into")
    crf: int = Field(
        22, ge=0, le=51, description="H.264 quality (lower = better); 26 erased faint objects"
    )
    clean_copy: list[str] = Field(
        ["Meteor"],
        description="Classes that also get a clip without markings, saved at once. Live "
        "nights need this (they aren't recorded); recordings can make one on request",
    )


class StorageConfig(_Section):
    """Where the station keeps its record of every night (see storage.py)."""

    database: str = Field("data/skyseer.db", description="SQLite file: sessions, events, reviews")
    runs_dir: str = Field("data/runs", description="Analysed recordings (one folder each)")
    live_dir: str = Field("data/live", description="Live camera nights (one folder each)")


class CameraConfig(_Section):
    """The live camera. Exposure, gain and colour apply to the Raspberry Pi camera."""

    width: int | None = Field(None, description="Frame width (None = the camera's default)")
    height: int | None = Field(None, description="Frame height (None = the camera's default)")
    fps: float = Field(10.0, gt=0, description="Frames per second; exposure fills each frame")
    exposure_ms: float | None = Field(
        None, gt=0, description="Exposure per frame (None = 95% of the frame time)"
    )
    gain: float = Field(8.0, ge=1, description="Sensor gain: higher = brighter, noisier")
    colour_gains: tuple[float, float] = Field(
        (2.0, 1.8), description="Fixed white balance (red, blue)"
    )


class GpsConfig(_Section):
    """Where the station is and the true time, from a GPS receiver (see gps.py)."""

    source: str = Field(
        "none", description="none, gpsd (the Linux GPS service) or a device like /dev/ttyACM0"
    )
    gpsd_host: str = "127.0.0.1"
    gpsd_port: int = 2947
    cache_file: str = Field(
        "data/station_location.json", description="Last good position, used until a new fix"
    )
    min_satellites: int = Field(4, ge=3, description="Satellites needed to trust a fix")
    max_clock_error_s: float = Field(
        2.0, gt=0, description="Warn when the system clock is this far from GPS time"
    )


class StationConfig(_Section):
    """The unattended station (`skyseer station`): when to observe, how to keep the disk."""

    camera: str = Field("0", description="Camera index, stream URL, or 'picamera' on a Pi")
    dark_sun_altitude: float = Field(
        -12.0,
        le=0,
        description="Observe while the Sun is below this (-6 civil, -12 nautical, -18 astro)",
    )
    always_on: bool = Field(False, description="Observe around the clock (e.g. for testing)")
    keep_clips_days: float = Field(
        30.0, gt=0, description="Clips of live nights older than this are deleted"
    )
    min_free_gb: float = Field(
        5.0, ge=0, description="Below this much free disk, the oldest clips are deleted first"
    )
    check_interval_s: float = Field(60.0, gt=0, description="How often to check the Sun")
    retry_s: float = Field(60.0, gt=0, description="Wait before retrying a failed camera")


class PipelineConfig(_Section):
    frame_skip: int = Field(1, ge=1, description="Process every Nth frame")
    disturbance_fraction: float = Field(
        0.005,
        description="If more than this fraction of the picture changes at once, the camera "
        "moved or the scene changed: those frames are ignored, not tracked",
    )
    disturbance_padding_s: float = Field(
        1.0, description="Tracks within this many seconds of a disturbance are discarded"
    )


class Config(_Section):
    pipeline: PipelineConfig = PipelineConfig()
    stabilise: StabiliseConfig = StabiliseConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    filter: FilterConfig = FilterConfig()
    star_filter: StarFilterConfig = StarFilterConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    rules: RulesConfig = RulesConfig()
    observer: ObserverConfig = ObserverConfig()
    astro: AstroConfig = AstroConfig()
    satellites: SatelliteConfig = SatelliteConfig()
    aircraft: AircraftConfig = AircraftConfig()
    showers: ShowerConfig = ShowerConfig()
    photometry: PhotometryConfig = PhotometryConfig()
    alerts: AlertConfig = AlertConfig()
    clips: ClipConfig = ClipConfig()
    storage: StorageConfig = StorageConfig()
    station: StationConfig = StationConfig()
    gps: GpsConfig = GpsConfig()
    camera: CameraConfig = CameraConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load defaults, then a YAML file (if given), then keyword overrides like
    ``load_config(detector={"sensitivity": 7})``."""
    data: dict[str, Any] = Config().model_dump()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    data = _deep_merge(data, overrides)
    return Config.model_validate(data)
