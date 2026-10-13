---
name: simulate
description: Create realistic SkySeer test videos with known truth, either real satellites crossing a camera's view at a chosen place and time (from current orbit data) or meteors from a chosen shower plus sporadics, then analyse them and score SkySeer against the truth. Use this whenever the user wants to test an idea without waiting for a clear night, asks "what would SkySeer see from X on date Y", "simulate the Orionids", "make a test video", wants to demonstrate satellite naming or shower identification, or needs evidence that a change improves accuracy.
---

# Simulate a scenario

Real skies are rare and unlabelled; simulations give exact truth on demand. The
simulator places real catalogue stars, real satellite orbits (SGP4) and physically
consistent meteors (radiant geometry, entry speed, brightness curve) into a video that
then goes through the normal pipeline.

## Satellites
Pick a time **1-2 hours after sunset** (or before dawn): satellites are then sunlit
while the observer is in darkness. Good directions are 40-60° up, often toward the west
in the evening.
```
.venv/Scripts/python.exe tools/simulate_scenario.py satellites --date 2026-10-14T19:45:00 --lat 40.42 --lon -3.70 --alt 50 --az 250 --count 5 --out data/sim/sats.mp4
```
If it says no satellites cross the view, shift the time or direction. Orbit data comes
from the cached CelesTrak catalogue (`skyseer orbits` refreshes it), so dates within
about a week of the catalogue's age are realistic.

## Meteors
Check which showers are active and where their radiants are:
```
.venv/Scripts/skyseer.exe showers --date 2026-10-21T23:00:00 --lat 40.42 --lon -3.70
```
Point the camera 30-80° away from the radiant (that's where meteors are long and
visible), with the radiant above the horizon:
```
.venv/Scripts/python.exe tools/simulate_scenario.py meteors --shower ORI --count 6 --date 2026-10-21T23:30:00 --lat 40.42 --lon -3.70 --alt 45 --az 110 --out data/sim/orionids.mp4
```
Every third meteor is sporadic, a built-in control for false shower matches.

## Analyse and score
The simulator prints the exact `skyseer analyze ...` command. Run it, then compare the
results table with `<video>.truth.json`:
- **Satellites:** every truth satellite named with the correct NORAD ID; clock offset
  ≈ 0 (or ≈ the offset you deliberately introduced by shifting `--start-time`)
- **Meteors:** correct shower or "sporadic"; magnitudes within ~0.3 of `peak_magnitude`
- **Missed objects:** check their brightness, since very faint fast meteors (around +3)
  are below the detection limit by physics, not by bug

## Be honest about what simulations prove
The simulator uses the same astronomy as SkySeer, so a pass shows the **pipeline**
works end to end. It does not independently verify the astronomy formulas. Those are
checked against external references in the tests (Skyfield, ESA tetra3 sample images,
IAU shower values, equinox dates). Say this when presenting results, especially to
recruiters.
