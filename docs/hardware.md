# Station hardware (planned build)

**Status:** not built yet (planned for December 2026). The software side is ready and
tested without the hardware (camera driver against a stand-in, GPS against recorded
messages, the scheduler with a simulated clock). The plan for the first nights, and the
measurements they should produce: [first-light.md](first-light.md).

**Prices** are rough euro figures from October 2026; check current ones. Raspberry Pi
raised its prices three times between December 2025 and April 2026 (memory chips got
dearer): the Pi 5 4 GB went from $60 to about $110.

## The budget kit (target: under €200)

| Part | Why | Approx. € |
|---|---|---|
| Raspberry Pi 5, **2 GB** | runs the whole pipeline. The station peaks at ~0.9 GB with the Pi profile's 384 MB clip buffer (measured on a PC at 2028 x 1520 with the orbit catalogue and star database loaded), leaving room on a 2 GB Pi | 60-75 |
| Active cooler | continuous processing inside a closed box | 6 |
| 64 GB microSD card (A2) | the system and a month of clips (old clips are deleted automatically) | 10 |
| Raspberry Pi HQ Camera (Sony IMX477, CS mount) | large pixels, manual exposure, good at night | 58-80 |
| 6 mm f/1.2 CS-mount lens | about 55° x 43° of sky (see below); f/1.2 gathers ~3x the light of an f/2.2 lens | 21-30 |
| Pi 5 camera cable, 300-500 mm | the Pi 5 has a narrower camera socket than the cable cameras come with | 4 |
| USB-C power supply | a 5 V / 3 A charger is enough: the station uses no USB devices (the official 27 W supply is ~€13) | 0-13 |
| Weatherproof box with a clear window | home-made: an IP65 junction box with an acrylic dome or a flat window glued in | 10-15 |

Total roughly **€170-220** (with a charger you already have).

What the budget kit does without, and how SkySeer copes:

- **No GPS:** the Pi takes the time from the internet over Wi-Fi (good to a few hundredths
  of a second, plenty for satellite names), and you type the location into
  `configs/pi.yaml` once. The dashboard warns if the clock drifts.
- **No aircraft receiver:** aircraft are named from the free OpenSky Network over the
  internet (`aircraft: source: opensky` in the Pi profile).
- **No dew heater:** fine on dry nights. Dew or frost on the window ends the others early,
  so it is the first thing worth adding.

## Later upgrades (optional)

| Part | What it adds | Approx. € |
|---|---|---|
| 5 V dew heater strip around the window | keeps dew and frost off, so whole nights are usable | 10 |
| USB GPS receiver (u-blox, ideally with PPS) | location and exact time without internet | 15-25 |
| Raspberry Pi 5, 4 GB instead of 2 GB | more memory headroom (same speed) | +30-40 |
| NVMe SSD on an M.2 HAT+ | lasts longer than an SD card under nightly writing | 25-65 |
| USB ADS-B receiver (RTL-SDR) and 1090 MHz antenna | aircraft names offline and without OpenSky's daily limit (`source: readsb`) | 35-50 |
| PoE+ HAT and injector | one Ethernet cable for power and network | 25-40 |

## Field of view and resolution

The IMX477's active area is 6.29 x 4.71 mm. With a 6 mm lens:

- horizontal 2·atan(6.29 / 12) = **55°**, vertical 2·atan(4.71 / 12) = **43°**
- at the station's 2028 x 1520 binned resolution, about **1.8 arcminutes per pixel**

That fits SkySeer's star database (fields of view 20-70°). A wider lens covers more sky
(more meteors) but each pixel sees more sky (fainter objects lost in noise), and above 70°
the solver needs the wide-field database (`astro.database_min_fov/max_fov`). A faster lens
(lower f-number, e.g. f/1.2) gathers more light: the cheapest way to see fainter objects.

## Assembly notes

- **Pointing:** about 50-60° above the horizon, away from streetlights and the Moon's path.
  No need to level or aim precisely: the station measures its exact pointing from the stars
  every night.
- **Focus** at night on a bright star (deploy/pi/README.md), then lock the focus ring.
- **Dew heater (upgrade):** wrap it around the window's edge. With a 3 A charger the Pi
  limits its USB ports to 0.6 A, so power a heater from its own 5 V supply.
- **Cooling:** the Pi 5 throttles above ~80 °C; the dashboard shows its temperature.
  Leave an air gap or a vent on the shaded side of the box.
- **GPS antenna (upgrade)** with a view of the sky (inside the dome is fine).

## Software setup

See [deploy/pi/README.md](../deploy/pi/README.md): one install script, then the station
runs on its own and serves the dashboard at `http://skyseer.local:8000`.
