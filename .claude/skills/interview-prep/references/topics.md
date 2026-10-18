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
13. Phase 4: clips, shake correction, crossings, learned classifier
14. Phase 5: database, web API, live station, dashboard
15. Phase 6: the unattended Raspberry Pi station
16. Phase 7: v1.0 release, and what using it revealed

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
- **The recognised pattern** (tetra3 patch 5, `return_pattern`; `Calibration.pattern`):
  tetra3 tries groups of 4 of the brightest stars (8 -> 70 groups in the fast search, 14
  -> 1,001 widened). Each group's 6 distances, divided by the longest, give 5 ratios that
  don't change with zoom or rotation: its hash code. A catalogue pattern with the same
  code is paired star by star (both sorted by distance from the pattern's centre), the
  pointing follows, and it is accepted only if the whole catalogue then lines up beyond
  chance (false-match probability < 1e-3, corrected for database size). The first pattern
  that passes wins, not necessarily the "best". Madrid test sky: HIP 84345, 86032
  (Rasalhague), 86742, 80816, recognised on try 16 of 70. Test: each pattern star agrees
  with SkySeer's independent full-catalogue match (same HIP, < 0.5 px); option off gives
  identical results to the original code (12/12 solves).
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
- **Short satellite tracks** in noisy video are a mix of real fragments and junk; the
  Phase 4 learned model did not fix this on real footage (sim-to-real gap, see 13).
- **No extinction or colour terms** in photometry yet.
- **Single-station shower association** has ~5% chance matches; two stations would fix it.
- **Next phases:** Phase 7 is the v1.0 software release (Oct 2026); Phase 8 builds the Pi station (Dec 2026) and measures it.
- **Dashboard has no login:** it listens on 127.0.0.1 unless told otherwise; fine on a home
  network, would need authentication before exposing it to the internet.

## 13. Phase 4: clips, shake correction, crossings, learned classifier
- **Clips** (`clips.py`): memory-bounded frame buffer (full rate for 8 s, older frames
  thinned by powers of two: memory grows with the log of time covered). Per event: zoomed
  H.264 clip (PyAV), trail image masked to the event's own path, context image. Measured:
  CRF 26 erased a 3-sigma faint object (peak 0.4 of 12); CRF 22 + grain tuning kept it.
- **Shake correction** (`stabilise.py`), star-tracker style: coarse search (brightest
  pixel within 15 px) then sub-pixel centroids of ~60 stars; median shift. 0.05 px error,
  0.7 ms/frame; FFT phase correlation was 19 ms/frame (rejected). Holt smoothing leaves
  sidereal drift alone (max lag 0.55 px); sudden knocks are kept corrected. Results:
  4 px wobble 416 false positives -> 0; 25 px knock missed by pixel counting, now caught.
  Bug story: OpenCV 5.0 `phaseCorrelate` windows its input in place. Real-footage story:
  time-lapse stills made star centroids scatter 1-2 px, causing 16 false "camera moved"
  events in 15 min; fixed with a 2.5 px agreement tolerance and a 0.3 s confirmation.
- **Crossing objects** (`track.py` `_merged`): a blob inside two tracks' gates and 1.4x
  their usual size is skipped; both coast. No ID swaps before or after; path deviation
  at a crossing 0.68% -> 0.09%.
- **Learned classifier** (`classify/model.py`, `docs/classifier.md`): gradient-boosted
  trees + isotonic calibration, 24 scale-free features + rule shares, grouped CV, trained
  on 500 domain-randomised simulated nights. Held-out simulation: 69.5% -> 96.7% objects
  right, 59 -> 1 false alarms, ECE 0.015. Real footage (224 blind labels): 19/31 vs rules
  25/31 -> rules stay default. Diagnosis: real objects slower on screen than 95% of
  simulated ones (wide lens), repeated frames, zero sky noise from compression. Ablation:
  rule features add nothing. Top feature: detection rate. Next: orbit-matched satellites
  as automatic real labels (weak supervision).
- **Interview angle:** the most valuable result is the honest negative one: measuring the
  sim-to-real gap, diagnosing it with feature distributions, and not shipping a model that
  only looks good in simulation.

## 14. Phase 5: database, web API, live station, dashboard
- **Why not keep Streamlit:** a headless Pi station needs an API anyway; Streamlit re-runs
  the whole script on every click and can't push live events. FastAPI + a React/TypeScript
  dashboard served by SkySeer itself (no extra server on the Pi).
- **Storage** (`storage.py`): SQLite (one file, no server), WAL mode so the live thread
  writes while the dashboard reads. Run folders are self-describing (events.json, run.json,
  clips) and importable anywhere.
- **API** (`api/app.py`): sessions, filtered events, reviews, night report, CSV/JSON export,
  media confined to the run folder (path-traversal test), health, uploads as background
  jobs (`api/jobs.py`, one at a time so a Pi stays responsive), MJPEG preview, WebSocket feed.
- **Live station** (`live.py`): pipeline in a thread; each event stored, logged to
  events.jsonl (crash-safe) and pushed to dashboards via asyncio queues
  (`call_soon_threadsafe`). `ReplaySource` plays a video in real time as if live: demos and
  tests without hardware.
- **Closing the ML loop:** dashboard reviews and orbit-matched satellites from real sessions
  become training rows for `skyseer train` (`classify/dataset.station_rows`); simulated
  sessions are flagged and excluded.
- **Bugs found by testing in a real browser/server:** missing `websockets` package (the test
  client hid it; now a real-uvicorn test), cp1252 console crash under a service/pipe, a
  cached index.html showing an old build (now no-cache + hashed assets), a test leaking into
  the real database (fixture scope).
- **Design choices to defend:** hand-drawn SVG charts (small, fast on a Pi, on-brand) instead
  of a chart library; hash routing (no router dependency); SIMULATION tags everywhere for
  honesty.

## 15. Phase 6: the unattended Raspberry Pi station
- **Night schedule** (`astro/sun.py`, `station.py`): observes while the Sun is below -12°
  (nautical twilight). Offline Sun formula vs NASA JPL DE421: median 0.003°, max 0.012°
  over 500 random cases; dusk in Madrid on 21 Oct 2026 within 1.5 s of JPL. Tested with a
  fake clock (a day and night in seconds).
- **Disk housekeeping:** deletes clips of old live nights (> 30 days, or oldest first below
  5 GB free); keeps records; never touches analysed recordings.
- **GPS** (`gps.py`): gpsd or raw NMEA (GGA/RMC, checksums), no extra libraries; cached last
  fix for instant start after reboot; warns when the system clock is > 2 s off GPS time
  (the Pi has no clock battery). Bug story: two-digit NMEA years (94 -> 1994 not 2094).
- **Camera** (`pi_camera.py`): auto-exposure/white balance/noise reduction off; exposure
  fills the frame; Y plane only; sensor timestamps at mid-exposure. Tested against a
  stand-in for picamera2.
- **Deployment** (`deploy/pi/`): install script, venv with system site-packages (picamera2
  comes from apt), systemd service with Restart=always, chrony for GPS time.
- **Performance** (`docs/performance.md`): i7-7700 at 2028x1520, whole live station 23.8 fps
  full / 41.5 fps half resolution, same objects found. Meteor vs faint-satellite frame-rate
  trade-off: a 0.3 s meteor spans 4 frames at 10 fps (missed), so the profile uses 15 fps.
  Pi numbers are estimates until measured.
- **Why no tilt sensor:** plate solving already gives exact pointing every night.

## 16. Phase 7: v1.0 release, and what using it revealed
- **Release material:** README with a results table (each number says how it was measured),
  CHANGELOG, MIT licence, screenshots from a separate showcase database (simulated nights
  only, tagged SIMULATION; no third-party footage shown), hero GIF made from SkySeer's own
  annotated output of a simulated sky built from real stars and real orbits.
- **Bugs found by using the product as a stranger would** (making screenshots, following
  the quick start), each now with a test:
  - The event page never drew brightness curves: the frontend read `magnitudes`, the
    backend saves `light_curve`. A contract mismatch between TypeScript types and Python
    output that no single-side test could catch.
  - Live cards showed broken images: events were pushed to the WebSocket before the
    background clip writer had finished. Fix: publish from the write job's completion
    callback (a race between two threads).
  - The same satellite reported twice: noise split a faint blob into two pieces ~2 px
    apart and the tracker started a second track. Fix: group components on a dilated mask,
    measure on the real pixels (`detector.group_px`). Quick-start scene: 6 events -> 5 for
    5 satellites. Real YouTube footage: 620 -> 531 tracks; blind labels: rules 25 -> 27/31
    real objects right, junk called real 15 -> 13/193 (model unchanged at 19/31).
  - A fresh clone answered "Not Found" (the built dashboard is git-ignored): now a page
    says how to build it.
- **Interview angle:** dogfooding and the "stranger's first five minutes" are part of
  engineering; integration seams (frontend/backend, thread/thread) are where bugs hide.
- **Clips without markings** (`clips.py` `write_clean_clip`, `clips.clean_copy`): a user
  asked to hide the box and trail on a fine meteor. Design constraint that decided it: a
  live night is never recorded (only ~2 min of frames in memory), so a clean clip must be
  saved at event time, while a recording can make one later from the original video.
  Result: clean copies saved at once only for meteors (rare, the ones worth sharing); any
  other event of a recording made on request (<1 s) and cached. Refactor kept the marked
  clips pixel-identical (checked by decoding before/after).
