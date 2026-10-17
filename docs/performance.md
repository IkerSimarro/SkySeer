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
