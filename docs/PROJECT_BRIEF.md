# SkySeer: project brief

*A summary of the project, how it works, and where it is going. Written so it can be pasted
into another AI assistant or shared with someone new to the project.*

## What it is

SkySeer is an automated night-sky monitoring system. A camera watches the sky. SkySeer
detects anything that moves, tells **satellites, meteors and aircraft** apart, works out
exactly **where in the sky** each one was, and **identifies** it: it names the specific
satellite from public orbit data (e.g. "STARLINK-4012, 550 km up") and the meteor
shower a meteor belongs to (e.g. "Perseid"). It works on recorded video today. Next month
it becomes a **physical station**: a Raspberry Pi 5 with a camera that runs all night,
on its own, outdoors.

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
   faint 1-2 px objects. Camera movement and scene changes are detected and ignored.
3. **Tracking:** Kalman-filter motion prediction plus optimal (Hungarian) assignment
   links detections into tracks. It works in streaming mode, so each event is reported as
   soon as the object leaves the view.
4. **Classification:** rules based on resolution-independent features (speed, straightness,
   aircraft strobe rhythm, cloud texture). A machine-learning classifier is planned.
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
8. **Output (today):** a command-line tool producing JSON/CSV reports, an annotated video
   with satellite names and predicted orbit lines, and calibration overlays.

## Measured results so far (simulations built from real star/orbit/shower data)

| Capability | Result |
|---|---|
| Camera pointing from stars | under 0.004° error; wrong solutions rejected by statistical checks |
| Satellite identification | 6/6 real satellites named out of 16,078; 0/300 false names |
| Camera clock error | measured to ±0.15 s |
| Detection / classification benchmark | 100% detection, 100% classification, 0 false alarms |
| Meteor shower identification | ~96% of shower meteors correct; ~5% of random meteors matched by chance |
| Meteor brightness | ±0.1 magnitude |

All tested against real data so far: ESA's tetra3 sample sky images, the user's own
footage (the sky position was solved), and an hour-long YouTube night-sky compilation
used for robustness testing.

## Technology

- **Language and tooling:** Python 3.11+, packaged with `uv`; ruff, mypy and pytest;
  GitHub Actions CI.
- **Libraries:** NumPy, SciPy, OpenCV, scikit-learn, Skyfield, sgp4, pydantic (typed YAML
  configuration), Typer + Rich (command-line tool).
- **Development machine:** Windows laptop.
- **Target device:** Raspberry Pi 5 + HQ camera (IMX477) + 6 mm lens + USB GPS.
- **Source:** public GitHub repository. The work lives on a feature branch and nothing
  is pushed without the author's approval.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Foundation: clean package, streaming engine, tests, CI | done |
| 1 | Star recognition / camera calibration | done |
| 2 | Satellite identification | done |
| 3 | Meteor science (showers, magnitudes, fireballs, light curves) | done |
| 4 | Smarter detection: ML classifier, one clip per event, camera-shake correction | next |
| 5 | FastAPI backend + custom "mission control" web dashboard (replacing an old Streamlit UI) | planned |
| 6 | Raspberry Pi station: camera, GPS, auto-start, enclosure, outdoor tests | needs hardware |
| 7 | Portfolio launch: README with measured results, demo video, v1.0 release | planned |

About 85% of the remaining work can be done before the hardware arrives.

## Working preferences

- **Plain-English explanations.** The author is a student and wants to understand
  every decision.
- **Phased, organised plans,** with honest results: what works, what doesn't, and why.
- **Measured claims only.** Every capability is validated with tests or simulations.
- **Safety first:** never give a confident wrong answer (wrong satellite name, false
  sky position). Refusing is better than guessing.

## What I'm looking for

Claude Code **skills** that would help with this project: repeatable workflows for
development, testing and validation, documentation and portfolio presentation, preparing
for interviews about the project, and later deploying to a Raspberry Pi.
