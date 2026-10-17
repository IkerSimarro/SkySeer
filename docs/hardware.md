# Station hardware (planned build)

**Status:** not built yet (planned for December 2026). The software side is ready and
tested without the hardware (camera driver against a stand-in, GPS against recorded
messages, the scheduler with a simulated clock). Prices are rough October 2026 figures;
check current ones.

## Parts

| Part | Why | Approx. € |
|---|---|---|
| Raspberry Pi 5, 8 GB | runs the whole pipeline; 8 GB leaves room for the clip buffer | 85 |
| Official 27 W USB-C power supply | the Pi 5 plus a dew heater need the headroom | 13 |
| Active cooler | continuous processing inside a closed box | 6 |
| NVMe SSD (128 GB+) on an M.2 HAT+, or a 64 GB A2 microSD | clips are written all night; SD cards wear out faster | 15-65 |
| Raspberry Pi HQ Camera (Sony IMX477, 12.3 MP, CS mount) | large 1.55 µm pixels, manual exposure, good at night | 55 |
| 6 mm CS-mount lens | about 55° x 43° of sky (see below) | 30 |
| Pi 5 camera cable, 300-500 mm | the Pi 5 uses the narrow 22-pin connector | 5 |
| USB GPS receiver (u-blox 7/8, ideally with a PPS output) | location and exact time without internet | 15-25 |
| Weatherproof box (IP65) with a clear acrylic or glass dome | rain and dust | 25 |
| 5 V dew heater strip around the dome | stops dew and frost fogging the view, which ends most nights early otherwise | 10 |
| Optional: PoE+ HAT + PoE switch/injector | one Ethernet cable for power and network | 25-40 |

Total roughly **€250-350**.

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
- **Dew heater:** wrap it around the dome's edge, powered from the Pi's USB (or a separate
  5 V supply if it draws more than ~1 A).
- **Cooling:** the Pi 5 throttles above ~80 °C; the dashboard shows its temperature.
  Leave an air gap or a vent on the shaded side of the box.
- **GPS antenna** with a view of the sky (inside the dome is fine).

## Software setup

See [deploy/pi/README.md](../deploy/pi/README.md): one install script, then the station
runs on its own and serves the dashboard at `http://skyseer.local:8000`.
