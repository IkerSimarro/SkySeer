# Changelog

What changed, phase by phase, with the measured effect of each change. Dates are when the
work was done. Full detail is in the commit messages.

## [1.0.0] - unreleased

The first complete release: SkySeer rebuilt from a Streamlit prototype into a tested,
installable system, with an astronomy layer, a web dashboard and the software for an
unattended Raspberry Pi station.

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
