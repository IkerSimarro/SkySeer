# SkySeer: project brief

*A summary of the project, how it works, and where it is going. Written so it can be pasted
into another AI assistant or shared with someone new to the project.*

## What it is

SkySeer is an automated night-sky monitoring system. A camera watches the sky. SkySeer
detects anything that moves, tells **satellites, meteors and aircraft** apart, works out
exactly **where in the sky** each one was, and **identifies** it: it names the specific
satellite from public orbit data (e.g. "STARLINK-4012, 550 km up") and the meteor
shower a meteor belongs to (e.g. "Perseid"). It works on recorded video today, and its
station software is ready: in December 2026 it becomes a **physical station**, a
Raspberry Pi 5 with a camera that runs all night, on its own, outdoors.

## Why it exists (the real goal)

It is a **portfolio project to get a job in the aerospace / space industry**. The
author is a computer-science student. Every decision is judged partly by **what an
aerospace recruiter or engineer will think when they see it on GitHub**. That means:
- real, measurable engineering results
- honest documentation of limitations
- techniques used in the space industry (star trackers, orbit propagation, space
  situational awareness)
- clean, professional, tested code

## How it works (pipeline)

1. **Input:** a video file or a live camera (OpenCV, later the Raspberry Pi camera).
   Every frame carries a UTC timestamp.
2. **Detection:** background subtraction (OpenCV MOG2) finds moving blobs, including
   faint 1-2 px objects. Camera shake is measured from the stars, star-tracker style, and
   cancelled; real camera moves and scene changes are detected and ignored.
3. **Tracking:** Kalman-filter motion prediction plus optimal (Hungarian) assignment
   links detections into tracks, coasting through moments when two objects overlap. It
   works in streaming mode, so each event is reported as soon as the object leaves the
   view, with a zoomed video clip, a trail image and a context image.
4. **Classification:** rules based on resolution-independent features (speed, straightness,
   aircraft strobe rhythm, cloud texture) by default; an optional machine-learning model
   (gradient-boosted trees, calibrated probabilities) trained on realistic simulations.
5. **Star calibration (plate solving):** recognises the star pattern to find exactly
   where the camera points. It uses ESA's tetra3 star-tracker algorithm plus a custom
   camera model with lens distortion, and needs no location or time. Every pixel then
   maps to real sky coordinates (RA/Dec, altitude/azimuth).
6. **Satellite identification:** propagates the whole public satellite catalogue (16,000+
   objects from CelesTrak, SGP4 model) and matches each observed path. It also measures
   the camera's clock error.
7. **Meteor science:** shower identification against the IAU Meteor Data Center list
   (radiant geometry plus entry-speed check), brightness in real magnitudes calibrated
   against catalogue stars, fireball alerts, and periodic-brightness ("tumbling satellite")
   detection.
8. **Output:** a web dashboard (FastAPI backend + React/TypeScript frontend, served by
   SkySeer itself): event cards with trail images and clips, an all-sky map of where each
   object crossed, night reports with CSV/JSON export, a live page that shows the camera
   and new detections the moment they happen, browser uploads, and "is this right?"
   review buttons whose answers become real training data. Also a command-line tool,
   JSON/CSV reports and an annotated video.

## Measured results so far (simulations built from real star/orbit/shower data)

| Capability | Result |
|---|---|
| Camera pointing from stars | under 0.004° error; wrong solutions rejected by statistical checks |
| Satellite identification | 6/6 real satellites named out of 16,078; 0/300 false names |
| Camera clock error | measured to ±0.15 s |
| Detection / classification benchmark (clean scenes) | 100% detection, 100% classification, 0 false alarms |
| Camera shake (simulated wobble up to 6 px, 5 Hz) | 0 false alarms (416 without correction at 4 px) |
| Learned classifier, 150 held-out *difficult* simulated nights | 96.7% classified right vs 69.5% for the rules; 1 false alarm vs 59 |
| Learned classifier, real YouTube footage (224 blind by-eye labels) | behind the rules: 19/31 real objects right vs 27/31 (sim-to-real gap; rules stay default) |
| Noise linked into fake objects (real 1080p recording) | 467 → 71 in 10 minutes; all 23 real satellite passes kept; junk called real 13 → 7 of 193 |
| Aircraft identification (ADS-B, simulated flights) | 150/150 named among 20 others; 0/90 false names |
| Meteor shower identification | ~96% of shower meteors correct; ~5% of random meteors matched by chance |
| Meteor brightness | ±0.1 magnitude |
| Moon position (vs NASA JPL DE421) | median 0.002°, max 0.006°; fraction lit within 0.0001 |
| A 93%-lit Moon in view (simulated glare and lens ghost) | 6/6 satellites named, 0 false objects |
| Sun position for the night schedule (vs NASA JPL DE421) | median 0.003°, max 0.012° |
| Analysing a 1080p30 recording (Intel i7-7700) | 30-34 fps, about real time (was 12-19) |
| Live station throughput at 2028×1520 (Intel i7-7700) | 23.8 fps full, 41.5 fps half resolution (Pi 5 not yet measured) |

All tested against real data so far: ESA's tetra3 sample sky images, the user's own
footage (the sky position was solved), and an hour-long YouTube night-sky compilation
used for robustness testing.

## Technology

- **Language and tooling:** Python 3.11+, packaged with `uv`; ruff, mypy and pytest;
  GitHub Actions CI.
- **Libraries:** NumPy, SciPy, OpenCV, scikit-learn, Skyfield, sgp4, pydantic (typed YAML
  configuration), Typer + Rich (command-line tool).
- **Development machine:** Windows laptop.
- **Target device:** Raspberry Pi 5 (2 GB) + HQ camera (IMX477) + 6 mm f/1.2 lens, a
  budget kit of about €170-220 (GPS, dew heater and an aircraft receiver are later
  upgrades; time from the internet, aircraft from OpenSky).
- **Source:** public GitHub repository (v1.0.0 and v1.0.1 released in October 2026). CI
  tests every push on Windows and Linux, and on Raspberry Pi OS (trixie and bookworm)
  on 64-bit ARM with the real camera library.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Foundation: clean package, streaming engine, tests, CI | done |
| 1 | Star recognition / camera calibration | done |
| 2 | Satellite identification | done |
| 3 | Meteor science (showers, magnitudes, fireballs, light curves) | done |
| 4 | Smarter detection: ML classifier, one clip per event, camera-shake correction | done |
| 5 | FastAPI backend + custom "mission control" web dashboard (replacing an old Streamlit UI) | done |
| 6 | Station software: night schedule, GPS, Pi camera driver, install kit, auto-start | done |
| 7 | v1.0 software release: README with measured results, demo, CHANGELOG | done (Oct 2026) |
| 8 | Station build: Pi 5 + HQ camera outdoors, enclosure, first-light results, real throughput | Dec 2026 |

## Working preferences

- **Plain-English explanations.** The author is a student and wants to understand
  every decision.
- **Phased, organised plans,** with honest results: what works, what doesn't, and why.
- **Measured claims only.** Every capability is validated with tests or simulations.
- **Safety first:** never give a confident wrong answer (wrong satellite name, false
  sky position). Refusing is better than guessing.
