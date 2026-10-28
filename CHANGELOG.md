# Changelog

What changed, phase by phase, with the measured effect of each change. Dates are when the
work was done. Full detail is in the commit messages.

## [Unreleased]

### Ready for the station build (2026-10-28)
- **A night cut off by a power cut or crash is closed properly.** A running night
  records its latest frame time once a minute; when a station next starts, any night
  left open with no heartbeat for 5 minutes ends at that time (not days later at the
  next disk clean-up), gets its events.json and run.json, and shows **cut off** in
  the dashboard.
- **CI now reproduces the Raspberry Pi's software.** It runs on 64-bit ARM, with
  Raspberry Pi OS's own packages (including the real camera library, picamera2). It
  installs SkySeer with the Pi install script's own Python step and runs every test.
  It passes on trixie, which new Pi images ship (Python 3.13, OpenCV 5.0, in 13 min).
  The camera driver's eight night settings are checked against the real libcamera
  controls. Two problems the Pi would have hit in December were found and fixed:
  - On bookworm, pip's numpy 2 broke the camera library's JPEG code. The install now
    keeps numpy on the system's major version.
  - PyAV older than 14.1 (bookworm ships 12.3) can't turn phone videos upright, so
    SkySeer now requires 14.1.
- The Pi camera driver uses the frame size the camera reports after setup, in case
  libcamera adjusts the requested 2028x1520 (it warns if so).
- A first-light plan for the station build ([docs/first-light.md](docs/first-light.md)).

## [1.0.0] - 2026-10-28

The first complete release: SkySeer rebuilt from a Streamlit prototype into a tested,
installable system, with an astronomy layer, a web dashboard and the software for an
unattended Raspberry Pi station.

### Release checks (2026-10-28)
- The star catalogue download falls back to the Harvard mirror when the CDS archive in
  Strasbourg doesn't answer (it timed out on every try on 28 October), and every copy is
  checked against the catalogue's SHA-256 before use.
- CI posts each failed test's traceback as a public annotation (job logs need a GitHub
  sign-in), type-checks cleanly on Python 3.11 as well as 3.12, and uses the Node 24
  versions of its GitHub Actions.

### A tidier dashboard (2026-10-28)
- One session switcher in the top bar drives every page (Home, Events, Sky map, Night
  report) instead of a selector on each, remembered in the browser; links still open a
  given session.
- The Overview became two pages: **Home** (the chosen session's best moments as large
  pictures, its counts and latest detections) and **Sessions** (cards with pictures of
  each session's most striking events, search, filters, sorting, rename and remove,
  optionally deleting SkySeer's own files for it).
- Test sessions made by the simulators are hidden unless "Show test sessions" is ticked,
  including from "all sessions" views (`/api/events?simulated=false`).
- Event pages keep the clip, identification and key measurements in view; rule scores and
  the technical measurements are behind a **Details** switch.
- Works on a phone: the side menu becomes a row of links and event cards sit two a row.

### Faster analysis that keeps the computer usable (2026-10-27)
- A 1080p30 recording now analyses at about its own speed: 19.4 → 33.7 frames/s on a
  dark sky and 12.2 → 30.3 on a light-polluted one with many satellites (Intel
  i7-7700, clips included), so 4.2 h instead of 7.3 h, and 6.1 h instead of 15.8 h.
  The star filter no longer recomputes every track's motion for each comparison
  (27.1 → 4.3 ms a frame), tracking compares all tracks at once (13.8 → 5.1 ms), the
  background model runs on the graphics card when there is one (OpenCL), and video is
  read with PyAV on several threads straight to greyscale.
- Same results: on 5 minutes of real video the processor path gives exactly the same
  5,404 tracks as before; the graphics card's rounding changes 3%, nearly all noise,
  and all 24 long satellite passes are found either way.
- Dashboard analyses run in their own low-priority process on half the processor's
  threads (all 8 were no faster than 4), can be stopped and resumed (progress is saved
  every 5 minutes of video), and the list of analyses survives a restart. "Quick look"
  searches a half-size picture: 2.6x faster, but it found only 10 of 24 faint
  satellites on a bright sky, so it is not the default.
- Detecting on a smaller picture now measures positions on the full one: 0.08 px error
  on 1080p test scenes instead of 0.71 px (it matters for the Pi profile's half size).
- Noise linked into tracks is no longer reported as satellites and meteors. Three new
  tests: the points must lie where their neighbours put them (smooth motion), with no
  sudden jump between two spots, and a satellite or aircraft must be seen in 30% of its
  frames or 40 times. On a dark, heavily compressed recording: 467 → 71 false objects in
  10 minutes (the rest are sparse specks these tests can't separate from a faint
  satellite). All 23 real satellite passes of another recording are kept, and the blind
  by-eye labels give the same 27/31 real objects with 7 junk tracks called real, not 13.
  The benchmark stays at 100% / 100% / 0.
- Calibration and detection always read the same greyscale (on OpenCV-written videos the
  two kinds differed by 0.06 mag in zero point, shifting every magnitude).

### A station kit under budget (2026-10-21)
- `docs/hardware.md` now lists a budget kit of about €170-220 (Raspberry Pi prices rose
  three times since December 2025): a 2 GB Pi 5, the HQ camera and 6 mm f/1.2 lens, a
  microSD card and a home-made window box. GPS, dew heater, SSD, PoE and an aircraft
  receiver became optional upgrades.
- A 2 GB Pi is enough, measured: with the clip buffer at 384 MB the whole station
  (orbit catalogue and star database loaded, 2028 x 1520 video) peaked at 857 MB on a PC,
  against 1,074 MB with the old 768 MB buffer. Clips stay sharp: buffered frames are kept
  at ~60% resolution and clips are written 640 px wide.
- The Pi profile now expects no GPS (internet time, typed location) and names aircraft
  from OpenSky.

### Aircraft identification (2026-10-19)
- Aircraft are named from their ADS-B broadcasts (position, height, speed, flight
  number), matched to the observed path with the same sideways-and-timing test as
  satellites (now shared code). Broadcasts come from the free OpenSky Network (no
  hardware; 400 requests a day without an account, 4,000 with one) or a local readsb
  receiver.
- `skyseer aircraft now` lists what is overhead; `skyseer aircraft record` logs
  broadcasts while you film with any camera, for `skyseer analyze --aircraft LOG`.
  `serve`/`live`/`station` take `--aircraft opensky|readsb`: the station asks only while
  something crosses the view, keeps the night's broadcasts in `aircraft.jsonl`, and names
  a pass once broadcasts after it have arrived (updating the dashboard).
- A confident match makes the event a Plane (`label_source: adsb`), outranking the rules
  and any satellite. Event pages show flight, ICAO address, height, distance, speed and
  the camera clock error; night reports and the CSV list aircraft.
- Measured: direction from a broadcast within 1" of Skyfield; 150/150 simulated flights
  named among 20 others (broadcasts every 10 s; 149/150 every 5 s), clock error to 0.04 s
  median; 0 of 90 random paths named. Distant aircraft that barely move are matched on
  position along the path and report the clock as not measurable; paths shorter than
  0.5° aren't matched (their direction can't be checked). Simulated view built from 80 s
  of real OpenSky traffic over Madrid, camera 13° above the horizon: 5 of 10 aircraft
  named, none wrongly; the other 5 were 88-154 km away, moving under a pixel a second,
  and broke into pieces too short to check.

### The recognised star pattern (2026-10-18)
- Plate solving now records which 4-star pattern tetra3 recognised (the 4 image stars,
  their Hipparcos numbers, in the pairing the solver used) and how many patterns it tried
  first. It is kept on the calibration, written to `calibration.json` (`pattern`,
  `solver.patterns_tried`, `solver.search`) and drawn on the overlay as orange lines.
- tetra3 patch 5: an opt-in `return_pattern` option. With it off, 12 of 12 solves on
  random simulated skies gave identical results to the original code (11 solved, 1 failed
  the same way in both).
- On 3 simulated skies, every pattern star is one of the stars SkySeer's independent
  full-catalogue match found, at the same position (< 0.5 px) with the same Hipparcos
  number. The Madrid test sky was recognised on the 16th of 70 patterns of the fast
  search; a harder test sky needed the widened search, after 347 of 1,001.

### Phase 7: release preparation (2026-10-17)
- New README with measured results, a demo, dashboard screenshots and an honest
  limitations section; this CHANGELOG; an MIT licence file (the licence the package
  already declared).
- **Fixed, found while making the screenshots:**
  - Brightness curves never appeared on event pages: the dashboard read a field named
    `magnitudes`, but SkySeer saves it as `light_curve`.
  - New detections on the Live page sometimes showed a broken trail image: they were
    announced before their clip and images had been written. They are now announced once
    the files exist (new test).
  - Noise could break a faint satellite's blob into two pieces a pixel or two apart, and
    the tracker then reported the same satellite twice. Pieces up to 2 px apart are now
    one detection, still measured on the real pixels (`detector.group_px`, new test). In
    the README's quick-start scene: 6 events → 5 for 5 satellites. On the real YouTube
    footage: 89 fewer junk tracks (620 → 531); against the blind labels, real objects
    classified right 25 → 27 of 31 and junk reported as real 15 → 13 of 193.
  - `tools/score_labels.py --labelled-run` scores a new analysis against labels made on
    an earlier one, by matching each labelled track to the new event that follows it.
  - Long passes drew ~2,000 noisy brightness points; the curve now shows the median of
    each of 200 time bins. The night report's activity axis no longer repeats "1".
  - A fresh clone has no built dashboard (the build is git-ignored), and the start page
    answered "Not Found". It now explains how to build it.
- The Live page lists detections under the camera view instead of below the whole page.
- **Clips without markings:** a Markings / Clean switch on event pages hides the box,
  trail and title so a fine meteor can be watched unobstructed. Meteors get the clean copy
  at once (`clips.clean_copy`), because a live night isn't recorded and the frames are gone
  once the clip is written. For an analysed recording any event's clean clip is made on
  request from the original video (under a second for a meteor here), then kept. Marked
  clips, trail and context images are pixel-identical to before.

### Phase 6: the unattended station (2026-10-16)
- `skyseer station` observes from nautical dusk to dawn, every night, unattended. It
  retries a failed camera every minute and serves the dashboard throughout.
- An offline Sun position formula decides when it is dark. Against NASA JPL DE421 over
  500 random times and places: median error 0.003°, maximum 0.012°.
- Disk housekeeping: clips of old live nights are deleted after 30 days, or oldest first
  below 5 GB free. Event records are always kept.
- GPS position and time come from gpsd or a raw NMEA receiver, with checksums checked and
  the last fix cached for reboots. The station warns when the clock is more than 2 s off
  GPS time.
- Raspberry Pi HQ camera driver (picamera2): fixed night exposure, automatic adjustments
  off, frame times from the sensor's own timestamps. Tested against a stand-in for
  picamera2; not yet on real hardware.
- Pi profile (`configs/pi.yaml`), install script, systemd service and setup guide in
  `deploy/pi/`; parts list and field of view in `docs/hardware.md`.
- Throughput at the HQ camera's 2028×1520 on an Intel i7-7700: 23.8 frames/s at full and
  41.5 at half processing resolution, finding the same objects. A 0.3 s meteor was missed
  at 10 frames/s, so the Pi profile uses 15.

### Phase 5: database, web API and dashboard (2026-10-16)
- SQLite station database, a FastAPI backend (REST, CSV/JSON export, MJPEG preview,
  WebSocket live feed, background upload jobs) and a React + TypeScript dashboard served
  by SkySeer itself. The dashboard has a Live page, event pages, an all-sky map, night
  reports and an upload page.
- "Is this right?" review buttons. Reviews, and satellites matched to an orbit, become
  training data for `skyseer train`.
- Simulated recordings are tagged SIMULATION everywhere, so they can't be mistaken for
  real skies.
- The old Streamlit app was removed: the dashboard does everything it did.

### Phase 4: smarter detection (2026-10-14)
- A zoomed clip, trail image and context image for every event, from a memory-bounded
  frame buffer. At CRF 26 a faint object vanished from the clip; CRF 22 with grain tuning
  keeps it.
- Camera-shake correction measured from the stars: 0.05 px accuracy in 0.7 ms per frame.
  False alarms from a 4 px wobble fell from 416 to 0.
- Crossing objects keep straight tracks: path deviation at an X crossing 0.68% → 0.09%.
- An optional learned classifier: gradient-boosted trees with calibrated probabilities,
  trained on 500 domain-randomised simulated nights. It scored 96.7% vs 69.5% for the
  rules in held-out simulation, but 19/31 vs 25/31 on real footage labelled blind by eye,
  so the rules stay the default (`docs/classifier.md`).

### Phase 3: meteor science (2026-10-08)
- Meteor shower association against the IAU Meteor Data Center list (113 showers): ~96%
  of shower meteors identified, ~5% of sporadic meteors matched by chance (Monte Carlo
  over 2,000 meteors).
- Magnitudes calibrated from catalogue stars (within ±0.1 in simulation), fireball alerts,
  and a periodicity search that reveals tumbling satellites.

### Robustness on real footage (2026-10-07)
- An hour-long YouTube compilation exposed several weaknesses:
  - Blob size was measured as the area inside the outline, which is 0 for 1–2 px objects.
  - Scene changes produced false alarms.
  - Speeds were measured in pixels per frame, which depends on the camera.
  - Aircraft strobes and insects were confused with satellites and meteors.

  After the fixes: 4 clean meteors and 28 complete satellite passes, instead of 47 false
  "meteors" and 12 "satellites".

### Phase 2: satellite identification (2026-10-07)
- SGP4 propagation of the ~16,000-object CelesTrak catalogue in ~40 ms. Each observed
  path is matched against the predictions, and the camera's clock error is measured.
- 6/6 simulated satellites named; 0 of 300 random paths named; a 2 s clock error
  measured to within 0.15 s. Orbits too old for a recording are refused.

### Phase 1: star calibration (2026-10-07)
- Plate solving with ESA's tetra3 and the Hipparcos catalogue, a camera model with lens
  distortion, and guards against false solutions. Every pixel maps to RA/Dec and alt/az.
- Pointing within 15″ on 30 simulated skies; the author's own footage was solved from 80
  stars.

### Phase 0: foundation (2026-09-30)
- The code was rebuilt as an installable package with a streaming pipeline: frame
  sources with UTC timestamps, MOG2 detection, a Kalman tracker with optimal assignment,
  and rule-based classification per track.
- Added typed configuration, logging, a synthetic sky generator with ground truth, a
  benchmark, tests and CI.
- Build artifacts, videos and Replit state were removed from version control.

## Prototype (2025-10 to 2026-02)
- A Streamlit app built on Replit: MOG2 motion detection, hand-made track features and
  K-Means clustering, with charts and a downloadable report.
