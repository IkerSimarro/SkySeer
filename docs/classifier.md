# The learned classifier: method and honest results

SkySeer labels every track as Satellite, Meteor, Plane or Junk. The default is a set of
hand-written rules; this page describes the optional learned model, how it was tested,
and why it is **not** the default yet.

## Method

- **Model:** gradient-boosted decision trees (scikit-learn `HistGradientBoostingClassifier`)
  with isotonic probability calibration. Trees suit tabular measurements, train in
  seconds, need far less data than a neural network and run comfortably on a Raspberry Pi.
- **Inputs:** 24 *scale-free* track measurements (speed in image diagonals per second,
  signal-to-noise instead of raw brightness, the fraction of frames the object was seen,
  path straightness, strobe rhythm, sky texture...) plus the rules' opinion. Anything in
  raw pixels or 0-255 levels would tie the model to one camera.
- **Training data:** 500 randomly drawn *difficult* simulated nights (domain
  randomisation): resolutions 640x360 to 1280x720, 25/30 fps, lenses from narrow to very
  wide, sensor noise, JPEG compression, camera wobble, repeated frames, faint and partial
  objects, tumbling and flaring satellites, varied aircraft strobes, plus distractors:
  insects near the lens, twinkling stars, hot pixels and drifting clouds. 4,783 labelled
  tracks (tracks the star-field filter claims first are left out). Build with `tools/build_training_set.py`, train with `skyseer train`.
- **Testing:** 150 further simulated nights never used for training (tracks of one video
  never appear on both sides), the 20 clean benchmark scenes, and a real hour-long
  YouTube compilation labelled by eye.

## Results

| Test | Rules | Model |
|---|---|---|
| 150 difficult simulated nights: objects classified right | 69.5% | **96.7%** |
| ...false alarms | 59 | **1** |
| ...aircraft F1 score | 0.44 | **0.92** |
| ...probability calibration error (ECE) | - | **0.015** |
| 20 clean benchmark scenes | 100% | 100% |
| Real footage: real objects labelled right (31) | **25** | 19 |
| ...meteors (6) | 4 | **6** |
| ...satellites (24) | **21** | 13 |
| ...junk called a real object | 15 | **11** |

Real-footage labels: a random sample of 200 tracks plus every track either classifier
called a real object, labelled **blind** (from trail images, without seeing either
answer); 21 tracks too ambiguous to call were left out. Score with
`tools/score_labels.py`.

Further findings:

- **What the model uses most** (accuracy lost when a measurement is shuffled): the
  fraction of frames the object was seen (7.5 points: twinkling stars, hot pixels and
  strobing aircraft blink in and out), then path straightness, duration and speed.
- **Ablation:** without the rules' opinion as an input the model is just as good (97.2%
  vs 96.7%), so it learns from the measurements themselves, not from the rules.
- **Learning curve:** 10% of the training scenes gives 91.4%, 25% gives 93.7%, 50%
  gives 94.4% and all of them 96.7%: more data still helps, but slowly.

## Why the rules stay the default: the sim-to-real gap

The first model scored 97% in simulation but recognised 0 of 6 real meteors. Comparing
real and simulated measurements showed why: every real satellite, meteor and aircraft
moved slower on screen than 95% of the simulated ones (the YouTube camera has a wide
lens). Adding lens width and repeated frames to the simulator fixed the meteors (6/6),
but real satellites are still missed, because this video has properties outside anything
simulated: passes lasting 100 s, objects visible in only 14-47% of frames (many frames
are repeated copies) and a sky so heavily compressed that its noise is exactly zero.

Those are quirks of one heavily compressed compilation, and the station's own camera
will look different. Note also that the real labels guided the simulator fix, so the
real-footage numbers after it are somewhat optimistic for the model.

**Next step:** train on real data. Once the Raspberry Pi station records real nights,
every satellite matched to a known orbit becomes an automatically and correctly labelled
example (`classify.dataset.run_rows`), so the model can learn the station's real footage
without hand labelling. Until it beats the rules on real recordings, the rules stay the
default (`classifier: {method: rules}`).
