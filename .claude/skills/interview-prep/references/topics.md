# SkySeer topics: decisions, numbers, where to look

Numbers come from tests, benchmarks and documented experiments, so check the code if in
doubt (it evolves). Paths are relative to the repository root.

## Contents
1. Big picture and architecture
2. Detection
3. Tracking
4. Classification
5. Robustness on real footage
6. Star calibration (plate solving)
7. Coordinate frames and time
8. Satellite identification
9. Meteor showers
10. Photometry and light curves
11. Testing and validation strategy
12. Honest limitations and what's next

## 1. Big picture and architecture
- **What it is:** video in → detect → track → classify → calibrate from stars →
  identify satellites and meteor showers → reports.
  See `src/skyseer/pipeline.py` (`Pipeline`, `run`).
- **Streaming design:** each event is emitted when its object leaves the view (tracks
  end after missing for N seconds), never at end-of-video, so the same code serves files
  and a live camera.
- **Why not keep the old Streamlit app:** it re-ran the whole script on every click,
  and processing lived inside the UI. A station needs a headless service, an API and a
  dashboard (planned for Phase 5).
- **Configuration:** typed pydantic models in `src/skyseer/config.py`; the YAML file
  `configs/default.yaml` must equal the code defaults (a test enforces this).
- **Hardware plan:** Raspberry Pi 5 + HQ camera (IMX477) + 6 mm lens (~55° field of
  view) + USB GPS for location and time.

## 2. Detection (`src/skyseer/detect.py`)
- **Method:** OpenCV MOG2 background subtraction (a per-pixel statistical model of the
  background), then connected components.
- **Bug found on real footage:** blob size was measured as the area *inside the
  outline*, which is 0 for 1-2 px objects, and a 3×3 morphological opening erased them
  too. Faint satellites in low-resolution video were invisible. Fixed by counting pixels
  and removing the opening.
- **Brightness-weighted centroids** give sub-pixel positions, needed for astrometry.
- **Per-detection measurements:** contrast against a local sky ring (used for strobe
  detection), texture relative to frame noise (cloudiness), flux and saturation (used
  for magnitudes).

## 3. Tracking (`src/skyseer/track.py`)
- **Constant-velocity Kalman filter** per track, plus **Hungarian algorithm** (optimal
  assignment via `scipy.optimize.linear_sum_assignment`). It replaced the original greedy
  nearest-neighbour matcher, which swapped crossing objects and could give two detections
  the same ID.
- **Gating:** the matching window is tight (8 px) right after a detection and widens with
  the gap by the speed error the track could have built up. Unconfirmed tracks expire after
  0.3 s.
- **Gap bridging up to 3 s** for faint satellites that fade in and out. On the YouTube
  test it cut 37 fragments to 16 clean tracks in one 7-minute section.

## 4. Classification (`src/skyseer/classify/rules.py`, `features.py`)
- **Rule baseline** with resolution-independent features. Speed is measured in **image
  diagonals per second** (the original app used px/frame, so the same meteor counted as
  fast in 4K and slow at 640 px).
- **Aircraft strobes:** `flash_features` looks for regular flashes at 0.5-2.5 Hz
  (anti-collision lights flash 40-100 times per minute). Double flashes are merged and
  missed flashes tolerated. Irregular flicker from faint objects in compressed video is
  not counted (checked with light-curve plots on real footage).
- **Shape rules:** meteors must be straight (maximum deviation ≤ ~1.2%; curved insect or
  bat streaks measured ≥ 2.5%). Satellites and planes use the 80th-percentile deviation,
  which is robust to strobe-induced position jumps.
- **Cloud rejection:** surrounding texture divided by the frame's own noise (~1 on clear
  sky, > 3 inside clouds).
- **Planned for Phase 4:** a machine-learning classifier trained on synthetic data plus
  satellites labelled automatically by orbit matches.

## 5. Robustness on real footage
- **Test video:** a 58-minute, 640×360 YouTube compilation (unknown place and time) with
  scene changes, clouds, insects and heavy compression.
- **Camera movement / scene change:** when > 0.5% of the picture changes at once (clear
  sky, even cloudy, stays < 0.3%), those frames are ignored, overlapping tracks are
  discarded, the background is re-learned and calibration is invalidated. Bug found: a
  freshly reset background flags 100% of pixels, which kept re-triggering resets and
  dropped half the video. Fixed with a short settling period.
- **Before the fixes:** 47 "meteors" and 12 "satellites", essentially all false. **After:**
  4 meteors (all clean straight streaks), 28 complete long satellite passes, 29 scene
  changes ignored (983 false tracks discarded).

## 6. Star calibration / plate solving (`src/skyseer/astro/calibrate.py`, `camera.py`)
- **Lost-in-space solver:** ESA's **tetra3** (Apache-2.0, vendored with documented
  NumPy 2 / SciPy patches in `astro/_vendor/`) matches 4-star patterns against a database
  built from the **Hipparcos** catalogue (118k stars) for 20-70° fields of view. It needs
  no location, time or pointing hint, the same idea as a spacecraft star tracker.
- **Camera model:** rectilinear projection + radial distortion (k1, k2). Initial attitude
  from **Wahba's problem** solved with SVD (classic spacecraft attitude determination),
  refined by robust least squares, then iteratively matched against the full catalogue
  out to the frame corners.
- **False-solution guards:**
  - matched stars must far exceed the expected chance matches (`3·chance + 10`)
  - residual below 1.5 px
  - a physically possible lens (the distortion must not fold back on itself)
  - added after a real screenshot produced a confident false solution
- **Results:**
  - 30/30 simulated skies solved; pointing error ≤ 15 arcsec; worst error anywhere in the
    frame < 1 px
  - ESA's real sample photos match their recorded mount pointing within ~1°
  - the author's own video frame solved (Pegasus, 80 stars)
- **Hint-based re-calibration** in under a second, falling back to a full solve if the
  camera moved. Live auto-calibration runs in a background thread.
- **Star finder** rewritten for compressed video: automatic sky mask (excludes trees and
  roofs), robust noise estimate, compactness test.

## 7. Coordinate frames and time (`src/skyseer/astro/frames.py`)
- **ICRS** (J2000-like, fixed to the stars) vs **ENU** (East-North-Up, fixed to the
  ground). A fixed camera's calibration is stored in ENU because the stars drift 15
  arcsec/s through the view.
- **ICRS → ENU** uses precession-nutation and sidereal time (Skyfield), ignoring polar
  motion and aberration (< 21 arcsec, far below the ~100 arcsec/px pixel scale).
  Atmospheric refraction is applied (up to ~0.5° near the horizon).
- **Sidereal drift:** a star-referenced calibration stays valid over time by rotating
  with the sky. The observer terms cancel, so it works without a location.
- **Celestial-navigation demo:** from one screenshot (horizon + plate solve + planets) the
  date (31 Jan 2025 via Venus and Saturn), local time and latitude (~42°N, consistent
  with the US) were derived. Longitude needs a clock.

## 8. Satellite identification (`src/skyseer/astro/orbits.py`, `satellites.py`)
- **Catalogue:** CelesTrak "active" + "visual" groups (~16k objects), cached for offline
  use, refreshed politely (CelesTrak asks for at most one download per 2 hours).
- **SGP4 propagation** of the whole catalogue with `SatrecArray` (16k objects in ~40 ms).
  TEME → Earth-fixed frame via GMST. Agrees with Skyfield to within 10 arcsec.
- **Matching:**
  - **cross-track RMS** (sideways distance, insensitive to clock errors)
  - **timing offset**, which measures the camera's clock error
  - **timing spread**, which rejects a wrong speed
  - Starlink "trains" get reduced confidence unless the clock is good
- **Orbital elements** are only valid for about a week; old recordings need historical
  TLEs (e.g. from Space-Track), otherwise identification is refused.
- **Results:**
  - 6/6 real satellites named from 16,078
  - a 2 s clock error measured as −2.00 ± 0.03 s
  - 0/300 random paths falsely named
  - cross-track error ~25 arcsec

## 9. Meteor showers (`src/skyseer/astro/showers.py`)
- **Data:** IAU Meteor Data Center established showers (113, bundled with citation; built
  by `tools/build_showers.py`, keeping the solution with the most meteors per shower).
- **Solar longitude** (J2000) accurate to ~0.01°, checked against the 2026 equinoxes
  and solstices.
- **Association tests:**
  - the backward path passes within 2.5° (plus a path-length-dependent error) of the
    radiant
  - the meteor moves away from the radiant
  - the radiant is 2-90° away, above the horizon, and the shower is active
  - observed / expected angular speed is between 0.6 and 1.5
  - expected speed = v∞·sin(D) / range, with height ~80 + 0.43·v∞ km and
    v∞ = √(vg² + 125.4)
- **Monte Carlo** (2,000 meteors with 1° radiant and 15% speed scatter): **~96%** of
  shower meteors identified, **~5%** of sporadics matched by chance. A stricter setting
  (1.5°) gives 2.4% false matches but only 87% found. **A single camera can't do much
  better because it sees only a 2D path; two stations triangulating an orbit can.**
- **Simulated Perseid night:** every detected meteor correctly assigned (Perseid or
  sporadic).

## 10. Photometry and light curves (`astro/photometry.py`, `astro/lightcurve.py`)
- **Zero point** from catalogue stars in the calibration image (4 px aperture, sky
  annulus, robust median), then m = ZP − 2.5·log10(flux). Saturated detections give only
  an upper limit.
- **Peak magnitude** taken from a 3-sample running median, so a single noise spike can't
  inflate it. Accuracy **±0.1 mag** in simulation (two biases were found and fixed:
  aperture size and peak noise).
- **Limiting magnitude:** a fast +3 meteor was below the detection limit in the
  simulated camera, a physical limit.
- **Fireball alerts** (mag ≤ −4): log file, plus an optional webhook or ntfy push. Opt-in,
  sent in the background, never blocking detection.
- **Tumbling detection:** Lomb-Scargle periodogram with a false-alarm probability and a
  minimum amplitude.

## 11. Testing and validation strategy
- **Synthetic scenes with known truth:** `src/skyseer/synth.py` (pixel-level) and
  `src/skyseer/astro/simulate.py` (real stars, real orbits, physical meteors).
- **Benchmark:** `skyseer bench` and `tools/validate.py` with the accepted numbers in
  `tools/baseline.json` (100% detection, 100% classification, 0 false positives on 20
  scenes).
- **Tests:** about 65 tests in `tests/`, including camera bump, 1-2 px objects, strobes,
  random-dot skies rejected, and a frozen orbit snapshot for repeatability. CI on
  Windows and Linux with ruff, mypy and pytest.
- **Answering "your simulator just agrees with your maths":** real-data checks exist
  (ESA images, own footage solve, Skyfield cross-check, IAU and published shower values,
  equinox dates). The remaining gap is real footage with ground truth, planned with the
  Orionids and the Pi.

## 12. Honest limitations and what's next
- **Real-footage validation** is still limited; most numbers come from simulation.
- **Short satellite tracks** in noisy video are a mix of real fragments and junk (Phase 4
  machine learning).
- **No extinction or colour terms** in photometry yet.
- **Single-station shower association** has ~5% chance matches; two stations would fix it.
- **Next phases:** Phase 4 (ML classifier, per-event clips, shake correction), Phase 5
  (FastAPI + dashboard), Phase 6 (Pi station), Phase 7 (portfolio launch).
