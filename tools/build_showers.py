"""Build SkySeer's bundled meteor-shower table from the IAU Meteor Data Center list.

Source: IAU MDC "established showers" file (streamestablisheddata<year>.txt) from
https://www.ta3.sk/IAUC22DB/MDC2022/ . Many showers have several published solutions;
we keep, per shower, the solution based on the most meteors (largest N) that has a radiant
and velocity. Its own activity window is used; if it reports none, the median window of the
other solutions (the union of all studies would make some showers last for months, which
invites chance associations).

Usage:  python tools/build_showers.py data/astro/iau_mdc_established_2026.txt
Writes: src/skyseer/astro/data/showers.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _num(s: str) -> float | None:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.startswith('"'):
            continue
        f = [x.strip().strip('"') for x in line.split("|")]
        if len(f) < 29:
            continue
        rows.append(
            {
                "iau_no": int(f[1]),
                "code": f[3].strip(),
                "status": int(f[4]),
                "name": f[6].strip(),
                "activity": f[7].strip(),
                "los_begin": _num(f[8]),
                "los_end": _num(f[9]),
                "los": _num(f[10]),
                "ra": _num(f[11]),
                "dec": _num(f[12]),
                "dra": _num(f[13]),
                "ddec": _num(f[14]),
                "vg": _num(f[15]),
                "n": int(f[28]) if f[28].strip().isdigit() else 0,
                "parent": f[31].strip(),
            }
        )
    return rows


def build(rows: list[dict]) -> list[dict]:
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] == 1 and None not in (r["los"], r["ra"], r["dec"], r["vg"]):
            by_code[r["code"]].append(r)
    showers = []
    for code, sols in sorted(by_code.items()):
        best = max(sols, key=lambda r: (r["dra"] is not None, r["n"]))
        windows = [
            (r["los_begin"], r["los_end"])
            for r in sols
            if r["los_begin"] is not None and r["los_end"] is not None
        ]
        if best["los_begin"] is not None and best["los_end"] is not None:
            begins, ends = [best["los_begin"]], [best["los_end"]]
        elif windows:
            begins = [statistics.median(w[0] for w in windows)]
            ends = [statistics.median(w[1] for w in windows)]
        else:
            begins, ends = [], []
        drift = [r for r in sols if r["dra"] is not None]
        ref = max(drift, key=lambda r: r["n"]) if drift else None
        showers.append(
            {
                "code": code,
                "iau_no": best["iau_no"],
                "name": best["name"],
                "solar_longitude_peak": best["los"],
                # Activity window; if no solution reports one, assume +-5 deg (about +-5 days).
                "solar_longitude_begin": begins[0] if begins else best["los"] - 5.0,
                "solar_longitude_end": ends[0] if ends else best["los"] + 5.0,
                "ra": best["ra"],
                "dec": best["dec"],
                "dra_per_deg": ref["dra"] if ref else 0.0,
                "ddec_per_deg": ref["ddec"] if ref and ref["ddec"] is not None else 0.0,
                "vg_km_s": best["vg"],
                "members": best["n"],
                "parent": best["parent"],
            }
        )
    return showers


def main() -> None:
    src = Path(sys.argv[1])
    showers = build(parse(src))
    out = Path(__file__).parents[1] / "src" / "skyseer" / "astro" / "data" / "showers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": "IAU Meteor Data Center, established showers (" + src.name + ")",
                "url": "https://www.ta3.sk/IAUC22DB/MDC2022/",
                "citation": "Jenniskens et al. 2020, Planet. Space Sci. 182, 104821; "
                "Jopek & Kanuchova 2017, Planet. Space Sci. 143, 3-6",
                "frame": "J2000; solar longitudes in degrees; drift in degrees per degree of solar "
                "longitude (~per day); vg = geocentric velocity",
                "showers": showers,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"{len(showers)} established showers -> {out}")


if __name__ == "__main__":
    main()
