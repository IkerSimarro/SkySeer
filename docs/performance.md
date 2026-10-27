# Performance at the station's camera settings

**Status:** measured on a development PC; the Raspberry Pi 5 figures are estimates until
the station is built (December 2026). Re-measure on the Pi with:

```
python tools/perf_station.py --fps 15
```

## Measured (Intel Core i7-7700, 2017 desktop CPU)

A synthetic night at the HQ camera's binned mode, 2028 x 1520 pixels, 30 s long, with
the full pipeline, including camera-shake correction and video decoding:

| Processing resolution | Pipeline only | Whole live station (clips, preview, database) | Objects found / correct (6 in the scene) |
|---|---|---|---|
| full (2028 x 1520) | 27.0 fps | 23.8 fps | 5 / 5 |
| half (1014 x 760, `detector.process_scale: 0.5`) | 58.2 fps | 41.5 fps | 5 / 5 |

Half-resolution processing found the same objects with no false alarms on this scene.
Positions and brightness are still measured on the full-resolution image.

## What the Pi 5 needs

The station keeps up if it processes at least as many frames per second as the camera
delivers. The Pi 5 is roughly half as fast per core as this PC (an estimate, not a
measurement), so the whole live station should manage about **12 fps at full resolution
and 20 fps at half resolution**.

## Choosing the frame rate: the meteor trade-off

- **Lower frame rate, longer exposures** (10 fps, ~95 ms each): more light per frame, so
  fainter satellites show. But a short meteor gets few frames. In the test scene a
  0.3 s meteor spanned only 4 frames at 10 fps and was rejected by the rule that
  separates objects from noise (at least 3 detections over 0.3 s). It was missed at both
  resolutions.
- **Higher frame rate** (25-30 fps, as many meteor networks use): more frames per meteor,
  but each frame is fainter and there are more frames to process.

`configs/pi.yaml` therefore starts at **15 fps with half-resolution processing**: the
estimated ~20 fps throughput leaves about 30% headroom, and a 0.4 s meteor gets 6 frames.
The Live page warns if processing ever falls behind the camera (frames being dropped).
The first real nights should settle it: check the Pi's measured fps (Live page) and
whether short meteors are caught, then adjust `camera.fps` and `detector.process_scale`.

## Analysing recordings on a PC (October 2026)

Measured on the same PC (Intel Core i7-7700, 8 threads; NVIDIA RTX 2060) with two
third-party 1080p30 night-sky recordings compressed for streaming: a dark sky (4.7 h) and
a light-polluted one crossed by many satellites (6.4 h). Whole pipeline, clips included,
2 minutes of each.

| Recording | Before | Now ("Full detail") | "Quick look" |
|---|---|---|---|
| Dark sky, 4.7 h | 19.4 fps (the video would take ~7.3 h) | 33.7 fps (~4.2 h) | not measured |
| Light-polluted, 6.4 h | 12.2 fps (~15.8 h) | 30.3 fps (~6.1 h) | 77.3 fps (~2.4 h) |

Analyses now run at a little more than the video's own speed. Where the time went on the
light-polluted sky, per frame:

| Step | Before | Now | What changed |
|---|---|---|---|
| Star filter and classification | 27.1 ms | 4.3 ms | each track's motion is remembered, not recomputed for every comparison (it was recomputed 1,002,324 times for 1,218 tracks in 30 s of video) |
| Motion detection | 29.4 ms | 16.1 ms | the background model runs on the graphics card (OpenCL) when there is one; less overhead measuring each blob |
| Tracking | 13.8 ms | 5.1 ms | all tracks against all detections at once |
| Reading the video | 4.5 ms | 2.5 ms | PyAV (FFmpeg) on several threads, straight to greyscale |
| Clip buffer | 4.2 ms | 1.0 ms | no colour conversion; bilinear shrinking (same picture to 56-60 dB PSNR) |

**Same results.** On 5 minutes of the light-polluted recording, the new code on the
processor gave exactly the same 5,404 tracks as the old one. On the graphics card the
background model rounds slightly differently (0.6 of 2 million pixels per frame), which
changed 3% of the tracks, nearly all of them noise: all 24 long, steady satellite passes
were found either way.

**Keeping the computer usable.** Dashboard analyses run in their own process at low
priority (Windows "below normal", Linux niceness 10) on half the processor's threads:
on this PC all 8 threads were no faster than 4 (28.7 against 30.3 fps). An analysis
saves its progress every 5 minutes of video, so a stopped one, or one the server was
running when it closed, carries on from there.

**What didn't work.** Detecting on a half-size picture for all HD video was the first
plan: 2.6-4.7x faster, and on the synthetic 1080p benchmark it found the same 59 of 60
objects. But on the light-polluted recording it found only 10 of the 24 long satellite
passes, because shrinking averages a faint point with its neighbours and loses more of
its light than of the noise. It is offered as "Quick look" instead, not as the default.
Positions are still measured on the full image whenever detection runs smaller (0.08 px
error against 0.71 px before), and meteor brightness reads about 0.03 magnitude brighter
on average at half size.

**A problem this found, and the fix.** On the dark recording, full-size detection
reported compression noise linked into tracks as real objects: 467 in a 10-minute sample
(all 24 checked by eye were noise). Three tests now ask whether a track is one object
moving or noise the tracker linked up: its points lie where their neighbours put them
(real satellite passes within 0.2-0.5 px, noise taken for meteors 8-21 px), it has no
sudden jump between two spots (two twinkling stars joined up), and a satellite or
aircraft is seen in at least 30% of its frames or 40 times. Result: 467 → 71 in the same
10 minutes, with all 23 real satellite passes of the light-polluted recording still
reported, and on the blind-labelled real footage the same 27 of 31 real objects right
with 7 junk tracks called real instead of 13 ([classifier.md](classifier.md)). The 71
left are still noise (12 of 12 checked): sparse specks that happen to line up, which
these measurements can't tell from a faint satellite seen only a few times.
