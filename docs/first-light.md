# First light: the December station build

The software is finished and tested without the hardware. This guide turns the
README's "planned", "estimate" and "not yet" into **measured, real-sky results**. Every
step says what to check and which number to write down. The numbers go into the table
at the end, and from there into the README.

It assumes the parts in [hardware.md](hardware.md) and the install guide in
[deploy/pi/README.md](../deploy/pi/README.md).

## Phase A: on the desk (daytime, about 2 hours)

1. **Assemble and install.** Flash **Raspberry Pi OS (64-bit)**, then follow
   deploy/pi/README.md sections 1-3. CI installs SkySeer the same way on 64-bit ARM
   with Raspberry Pi OS's own packages, so this should work first time.
   - Check `rpicam-hello --list-cameras` shows `imx477`.
   - Write down how long `install.sh` took.
2. **Run the test suite on the Pi itself.** This proves the code on the real machine,
   not just in CI.
   ```
   cd ~/skyseer
   .venv/bin/pip install -e . --group dev
   .venv/bin/python -m pytest -q
   ```
   Write down: tests passed, and how long they took.
3. **Measure the real throughput.** The figures in performance.md are estimates
   until this step. Stop the station first (`sudo systemctl stop skyseer`), then run:
   ```
   .venv/bin/python tools/perf_station.py --fps 15
   ```
   Write down the frames per second (full and half resolution) for the whole live
   station. The station keeps up if it beats the camera's 15 frames/s; the estimate
   was about 20. While it runs, a second terminal with `vcgencmd measure_temp` and
   `free -h` gives the temperature and memory (the estimate was ~0.9 GB peak).
4. Start the station again: `sudo systemctl start skyseer`.

## Phase B: the first clear night (about an hour of attention)

1. **Place it** outdoors, 50-60° above the horizon, away from streetlights. Put the
   location in `configs/pi.yaml` (deploy/pi/README.md section 3).
2. **Clock.** `chronyc tracking`: write down the "System time" offset. Satellite names
   need the clock right to within a few seconds; the internet keeps it to milliseconds.
3. **Focus** on a bright star on the Live page until it is the smallest dot, then lock
   the ring (deploy/pi/README.md section 6). This matters more than anything else.
4. **Exposure.** If the Live picture is grey and noisy, lower `camera.gain`; if only
   the brightest stars show, raise it. Check the Live page's frames per second stays
   above 15.
5. **Calibration.** Once enough stars are visible, the Live page changes from "waiting
   for enough stars" to the camera's pointing (altitude and azimuth). The full solution
   is in the night's `run.json` (`data/live/<night>/run.json`, under `calibration`)
   or at `http://skyseer.local:8000/api/sessions`. Write down:
   - `stars_matched` and `rms_px` (the residual: how well catalogue stars fit, in pixels);
   - `fov_h` and `fov_v`: the measured field of view, against the **55° x 43°**
     hardware.md predicts for a 6 mm lens.
6. Leave it to run until dawn.

## Phase C: the morning after (check the results by eye)

This is the most important step for the portfolio: **the first satellite named in real
footage**, checked against an independent source.

1. **Satellites.** Events page, satellites only. For 10 named passes, check each
   against [Heavens-Above](https://heavens-above.com) for your location and that night:
   - the time;
   - the direction it crossed;
   - and roughly its brightness.

   Write down: how many were named, how many you checked, how many were right, and
   any that were wrong (that's a bug report: keep the event).
2. **Clock error.** Each satellite event page shows the camera clock error SkySeer
   measured from the orbit match. Compare it with chrony's offset from Phase B.
3. **Meteors.** The Night report: how many meteors, and which showers they were
   assigned to.
4. **Review a few dozen events** with the dashboard's review buttons (right / wrong
   class). These labels become real training data for the classifier (`skyseer train`).
   They also give an honest false-alarm count.
5. **Did it run itself?** `journalctl -u skyseer --since yesterday`: it should start at
   dusk, stop at dawn and report no errors.

## Phase D: a week of nights (it runs itself)

- **Power-cut test.** Pull the plug once mid-night, then plug it back in. Check:
  - the station starts again;
  - the interrupted night shows **cut off** on the Sessions page, and ends at about
    the time you pulled the plug;
  - observing carries on in a new session.
- **Disk.** `du -sh ~/skyseer/data/live/*`: write down the gigabytes per night. Old
  clips are deleted after 30 days, or sooner if the disk runs low.
- **Temperature** inside the box on the warmest day (the Pi slows down above ~80 °C).
- **The Geminids**, the year's strongest meteor shower, peak on the night of
  **13-14 December 2026**. That night the Moon is a 21% crescent that sets at about
  21:50 Madrid time (20:50 UTC), so the rest of the night is dark. Write down the
  meteors per hour after midnight, and how many SkySeer assigned to the Geminids.

## What to write down

| Measurement | Where it comes from | Result |
|---|---|---|
| Install time | `install.sh` | __ min |
| Tests on the Pi | `pytest -q` | __ passed in __ min |
| Throughput, whole station at 2028x1520 | `tools/perf_station.py --fps 15` | __ frames/s full, __ half (estimate: ~20) |
| Peak memory, temperature | `free -h`, `vcgencmd measure_temp` | __ GB, __ °C |
| First plate solve | `run.json` → `calibration` | __ stars, __ px residual, FOV __° x __° (predicted 55° x 43°) |
| Clock offset | `chronyc tracking` | __ ms |
| Satellites named, first night | Events page | __ named; __ of __ checked correct (Heavens-Above) |
| Camera clock error from satellites | satellite event pages | __ s |
| Meteors | Night report | __ meteors, __ assigned to showers |
| Geminids peak night | Night report, 13-14 Dec | __ meteors/h, __% assigned to Geminids |
| Reviewed events | review buttons | __ reviewed, __ false alarms |
| Power-cut test | Sessions page | cut off at __, restarted: yes / no |
| Disk per night | `du -sh` | __ GB |

With these numbers, the README's results table and limitations can be rewritten from
real skies, using the `portfolio-update` workflow.

## If something goes wrong

The troubleshooting list in [deploy/pi/README.md](../deploy/pi/README.md) covers the
camera not starting, missing satellite names and an unreachable dashboard. For
anything else, keep the night's folder (`data/live/<night>/`) and the log
(`journalctl -u skyseer > log.txt`): together they hold everything needed to find the
cause.
