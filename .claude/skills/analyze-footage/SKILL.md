---
name: analyze-footage
description: Analyse a real night-sky video with SkySeer end to end, then honestly judge the results by eye. Covers inspecting the video, finding scene changes, attempting star calibration, running the analysis, and building trail-image galleries to verify satellites, meteors and planes, ending with a scorecard of what was right and wrong. Use this whenever the user brings a new video or recording (in samples/ or elsewhere), says "analyse this video", "test SkySeer on my footage", "I recorded the sky last night", "what's in this video", or wants to see SkySeer working on real data.
---

# Analyse real footage

Real footage is the true test of SkySeer, because simulations only prove what they model.
Treat each video as an investigation: run SkySeer, then check its claims against what is
really in the video, and report honestly. Improvements found this way are the most
valuable kind.

## 1. Get the video and its context
- Videos live in `samples/` (git-ignored). If the user doesn't know the path, the
  project folder is the repository root; create `samples/` if missing.
- Ask for what makes the astronomy possible, but don't block on it:
  - **recording start time** (with time zone; SkySeer wants UTC or an explicit offset,
    e.g. `2026-10-21T22:30:00+02:00`)
  - **location** (latitude/longitude, e.g. by right-clicking in Google Maps)
  - **camera and lens**, if known
  - **what they think is in it**, which is useful ground truth

## 2. Inspect
```
.venv/Scripts/python.exe tools/footage/inspect_video.py samples/VIDEO.mp4
```
This gives resolution, fps, duration, bitrate, **scene changes** (compilations and
moved cameras) and `overview.jpg`. Look at the overview image: clouds, trees, a moon glow,
the horizon. Low resolution or heavy compression predicts faint objects and weak star
recognition, so say so up front.

## 3. Calibrate (sky position)
```
.venv/Scripts/skyseer.exe calibrate samples/VIDEO.mp4 --at SECONDS [--time ... --lat ... --lon ...]
```
Try a couple of moments in a clear stretch (one per scene). If it fails, that's an
acceptable answer: SkySeer refuses rather than guesses. Common causes are too few stars,
a lens wider than the 20-70° database, or compression. Calibration needs the time and
location for altitude/azimuth and satellite names. Without them it still gives RA/Dec.

## 4. Analyse
```
.venv/Scripts/skyseer.exe analyze samples/VIDEO.mp4 [--start-time ... --lat ... --lon ...] [--no-sky] --include-junk -o data/runs/NAME
```
- Use `--no-sky` if calibration failed (it saves time and avoids a pointless retry).
- Long videos take minutes (~0.1 × real time on the dev laptop), so run them in the
  background.
- `--include-junk` keeps rejected tracks too, so false negatives can be studied.

## 5. Verify by eye (the important part)
```
# What SkySeer claims (one tile per event, every frame layered into one image):
.venv/Scripts/python.exe tools/footage/trail_gallery.py samples/VIDEO.mp4 data/runs/NAME/events.json data/runs/NAME/meteors_planes.jpg --labels Meteor,Plane
.venv/Scripts/python.exe tools/footage/trail_gallery.py samples/VIDEO.mp4 data/runs/NAME/events.json data/runs/NAME/satellites.jpg --labels Satellite --min-duration 3
# What was really there (everything that moved, minute by minute):
.venv/Scripts/python.exe tools/footage/minute_trails.py samples/VIDEO.mp4 data/runs/NAME/trails.jpg --from-min 0 --to-min 12
```
How to read trail images:

| Looks like | Probably |
|---|---|
| Smooth straight line, often long | Satellite |
| Dotted line with a regular rhythm | Aircraft strobe |
| Short, bright, perfectly straight streak | Meteor |
| Curved, hooked or wiggly bright streak | Insect or bat near the lens |
| Blobs or texture | Clouds, compression |

Compare the minute trails with the events to find **misses**. View the images yourself
before claiming anything, and send the most useful ones to the user.

## 6. Report: an honest scorecard
- A table of counts per class, plus what was verified as correct, wrong and missed, with
  tile numbers.
- **Patterns behind the errors.** Each recurring error type is a potential general fix
  (e.g. curved streaks taught us "meteors must be straight"). Propose fixes that would
  help any footage, not tweaks that only fit this video, and confirm with the user before
  changing behaviour.
- **What couldn't be judged** (e.g. planes vs. satellites without ground truth) and what
  information would settle it.
- If changes are made, run the `validate` skill afterwards: fixes for real footage
  must not break the benchmark.
