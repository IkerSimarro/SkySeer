# Running SkySeer on a Raspberry Pi station

The station is a Raspberry Pi 5 with the HQ camera looking up at the sky. It observes
every night from dusk to dawn on its own, names satellites, records meteors and serves
the dashboard on your home network. Parts list and build notes: [docs/hardware.md](../../docs/hardware.md).

## 1. Prepare the Pi

1. Flash **Raspberry Pi OS (64-bit, Bookworm)** with Raspberry Pi Imager. In its settings
   set a hostname (e.g. `skyseer`), your Wi-Fi and SSH.
2. Connect the HQ camera to a camera port (power off first) and boot. Check it is seen:
   ```
   rpicam-hello --list-cameras
   ```

## 2. Install

```
git clone https://github.com/IkerSimarro/SkySeer.git ~/skyseer
cd ~/skyseer
./deploy/pi/install.sh              # add --gps if you fitted a USB GPS receiver
```

The script installs the system packages, a Python environment, the dashboard, the star
database (several minutes the first time) and the satellite orbits, then starts the
**skyseer** service, which runs `skyseer station` at every boot and restarts it after a
crash or power cut.

Open the dashboard from any device on the same network: `http://skyseer.local:8000`.

## 3. Tell it where it is

The budget kit has no GPS: put the location in `configs/pi.yaml` (once):

```yaml
observer:
  latitude: 40.420     # degrees north (south is negative)
  longitude: -3.700    # degrees east (west is negative)
  elevation_m: 650
```

then `sudo systemctl restart skyseer`. Where the camera points is found from the stars
automatically each night (plate solving): no compass or level needed.

With a **GPS receiver** (an optional upgrade) nothing needs typing: install with
`--gps`, set `gps: source: gpsd` in `configs/pi.yaml`, and the station reads its position
(and the time) from it, remembering the last position for straight after a reboot.

## 4. Time

Satellite names need the clock right to within a few seconds. The Pi has no clock
battery, so after a power cut it relies on the internet or the GPS. With `--gps`,
`chrony` takes the time from the GPS (to ~0.1 s from its messages; to microseconds if the
receiver's PPS wire is connected to a GPIO pin, see the comment in
`/etc/chrony/conf.d/skyseer-gps.conf`). The dashboard warns if the clock and GPS disagree.

Check: `chronyc sources` (a `*` marks the source in use) and `cgps` (the GPS fix).

## 5. Naming aircraft

SkySeer names the aircraft it sees from the positions they broadcast (ADS-B). The Pi
profile already uses the free OpenSky Network (`aircraft: source: opensky`, needs
internet); set `source: none` to turn it off. Without an account OpenSky allows 400 requests a day; SkySeer only asks while something
crosses the view and stops at `max_requests_per_day`. A free account allows 4,000: put
its API client's id and secret in the service's environment as
`SKYSEER_OPENSKY_CLIENT_ID` and `SKYSEER_OPENSKY_CLIENT_SECRET`, and raise the limit. A
USB ADS-B receiver running readsb (`source: readsb`) works offline with no limits.

## 6. Focus and exposure

- **Focus on stars, at night.** Open the Live page, point the camera at a bright star
  and turn the lens focus ring until the star is the smallest, sharpest dot. Then lock
  the ring. Focus is the single biggest factor for faint satellites and for star
  calibration.
- **Exposure** is fixed (no automatic adjustment at night): 15 frames per second, each
  exposed for ~63 ms, gain 8. If the sky looks grey and noisy (light pollution, Moon),
  lower `camera.gain` in `configs/pi.yaml`; if only bright stars show, raise it. The
  frame rate is a trade-off between faint satellites and short meteors; see
  [docs/performance.md](../../docs/performance.md) and check the Live page's frames per
  second: it must stay above `camera.fps`.

## 7. Everyday use

| Want to... | Do |
|---|---|
| See what it saw | the dashboard: Home, Sessions, Events, Night report |
| Watch live | the Live page (from nautical dusk) |
| Check it is running | `systemctl status skyseer` |
| Read its log | `journalctl -u skyseer -f` |
| Update SkySeer | `cd ~/skyseer && git pull && .venv/bin/pip install -e . && sudo systemctl restart skyseer` |
| Stop it | `sudo systemctl stop skyseer` (and `disable` to stop it starting at boot) |

Disk space looks after itself: clips of live nights older than 30 days, or the oldest
first when less than 5 GB is free, are deleted (the event records are kept). Both limits
are in `configs/pi.yaml` (`station:`).

## Troubleshooting

- **"Could not start the camera"** on the Live page: check the ribbon cable and
  `rpicam-hello --list-cameras`; the station retries every minute by itself.
- **No satellite names**: the location is missing, or the clock is wrong (see 4), or the
  orbits are old (`skyseer orbits --update` with internet).
- **Dashboard unreachable**: try the Pi's IP address instead of `skyseer.local`; check
  `systemctl status skyseer`.

**Security:** the dashboard has no login. Keep the station on your home network; don't
forward its port to the internet.
