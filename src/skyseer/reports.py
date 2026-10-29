"""Reports built from event records: the CSV export and the night report.

Everything here works on event dictionaries (``Event.to_dict()``, as stored), so the
command line, the API and the dashboard all produce the same numbers.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REAL = ("Satellite", "Meteor", "Plane")

CSV_BASE = [
    "id",
    "label",
    "confidence",
    "start_time",
    "end_time",
    "duration_s",
    "first_frame",
    "last_frame",
]
CSV_SKY = [
    "angular_speed_deg_s",
    "arc_deg",
    "start_alt",
    "start_az",
    "end_alt",
    "end_az",
    "satellite",
    "norad_id",
    "clock_offset_s",
    "peak_mag",
    "shower",
    "aircraft",
    "icao24",
    "aircraft_altitude_m",
]


def name_of(e: dict[str, Any]) -> str | None:
    """What it was identified as: an aircraft's flight number, a satellite's name or a
    meteor shower's code."""
    plane = e.get("aircraft")
    if isinstance(plane, dict) and e.get("label") == "Plane":
        return str(plane.get("name") or plane.get("icao24", "")).strip() or None
    ident = e.get("identification")
    if isinstance(ident, dict):
        return str(ident.get("name", "")).strip() or None
    shower = e.get("shower")
    if isinstance(shower, dict):
        return str(shower.get("code"))
    if shower == "sporadic":
        return "sporadic"
    return None


def magnitude_of(e: dict[str, Any]) -> float | None:
    phot = e.get("photometry")
    return phot.get("peak_magnitude") if isinstance(phot, dict) else None


def events_csv(events: list[dict[str, Any]]) -> str:
    """One row per event: times, class, sky position, identification, then all features."""
    out = io.StringIO()
    feature_keys = list(events[0]["features"]) if events else []
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(CSV_BASE + CSV_SKY + feature_keys)
    for e in events:
        sky = e.get("sky") or {}
        ident = e.get("identification") or {}
        plane = e.get("aircraft") or {}
        shower = e.get("shower")
        start, end = sky.get("start", {}), sky.get("end", {})
        extra = [
            sky.get("angular_speed_deg_s", ""),
            sky.get("arc_deg", ""),
            start.get("alt", ""),
            start.get("az", ""),
            end.get("alt", ""),
            end.get("az", ""),
            str(ident.get("name", "")).strip(),
            ident.get("norad_id", ""),
            ident.get("clock_offset_s", ""),
            "" if magnitude_of(e) is None else magnitude_of(e),
            shower.get("code") if isinstance(shower, dict) else (shower or ""),
            plane.get("name", ""),
            plane.get("icao24", ""),
            plane.get("altitude_m", ""),
        ]
        writer.writerow(
            [e[k] for k in CSV_BASE] + extra + [e["features"].get(k, "") for k in feature_keys]
        )
    return out.getvalue()


def write_reports(events: list[dict[str, Any]], out_dir: Path) -> None:
    """events.json and events.csv in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (out_dir / "events.csv").write_text(events_csv(events), encoding="utf-8")


def summary(e: dict[str, Any]) -> dict[str, Any]:
    """The few fields an event list or card needs (no track points or features)."""
    sky = e.get("sky") or {}
    return {
        key: e.get(key)
        for key in (
            "id",
            "session_id",
            "event_id",
            "label",
            "label_source",
            "confidence",
            "review",
            "start_time",
            "end_time",
            "duration_s",
            "fireball",
            "moon",
            "media",
        )
    } | {
        "name": name_of(e),
        "magnitude": magnitude_of(e),
        "sky": {k: sky[k] for k in ("start", "end", "angular_speed_deg_s", "path") if k in sky}
        or None,
    }


def night_report(session: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of one session: what was seen, when, and how well the station worked."""
    real = [e for e in events if e["label"] in REAL]
    counts = Counter(e["label"] for e in real)
    times = sorted(datetime.fromisoformat(e["start_time"]) for e in events)
    start = datetime.fromisoformat(session["started"]) if session.get("started") else None
    start = start or (times[0] if times else None)
    end = datetime.fromisoformat(session["ended"]) if session.get("ended") else None
    end = end or (max(datetime.fromisoformat(e["end_time"]) for e in events) if events else start)
    timeline = _timeline(real, start, end)
    satellites = [
        {"id": e["id"], "time": e["start_time"], "name": name_of(e), "magnitude": magnitude_of(e)}
        for e in sorted(real, key=lambda e: e["start_time"])
        if e["label"] == "Satellite"
    ]
    meteors = sorted(
        (
            {
                "id": e["id"],
                "time": e["start_time"],
                "shower": name_of(e),
                "magnitude": magnitude_of(e),
                "fireball": bool(e.get("fireball")),
            }
            for e in real
            if e["label"] == "Meteor"
        ),
        key=lambda m: (m["magnitude"] is None, m["magnitude"] or 0.0),
    )
    showers = Counter(m["shower"] for m in meteors if m["shower"])
    aircraft = [
        {
            "id": e["id"],
            "time": e["start_time"],
            "name": name_of(e),
            "altitude_m": (e.get("aircraft") or {}).get("altitude_m"),
        }
        for e in sorted(real, key=lambda e: e["start_time"])
        if e["label"] == "Plane"
    ]
    meta = session.get("meta") or {}
    return {
        "session": {
            k: session.get(k) for k in ("id", "name", "title", "kind", "source", "started")
        },
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "duration_s": round((end - start).total_seconds(), 1) if start and end else 0.0,
        "counts": {label: counts.get(label, 0) for label in REAL},
        "rejected": sum(e["label"] in ("Junk", "Star") for e in events),
        "reviewed": sum(e.get("review") is not None for e in events),
        "identified_satellites": sum(1 for s in satellites if s["name"]),
        "identified_aircraft": sum(1 for a in aircraft if a["name"]),
        "timeline": timeline,
        "satellites": satellites,
        "meteors": meteors,
        "aircraft": aircraft,
        "showers": dict(showers.most_common()),
        "brightest_meteor": meteors[0] if meteors and meteors[0]["magnitude"] is not None else None,
        "observer": meta.get("observer"),
        "pointing": meta.get("pointing"),
        "time_source": meta.get("time_source"),  # given / video file / unknown (recordings)
        "disturbances": meta.get("disturbances", []),
        "moon": meta.get("moon"),  # phase, hours up, minutes in view (astro.moon)
        "near_moon": sum(1 for e in real if e.get("moon")),
    }


def _timeline(
    events: list[dict[str, Any]], start: datetime | None, end: datetime | None
) -> dict[str, Any]:
    """Counts per class in about 24 equal time bins (at least 5 seconds each)."""
    if start is None or end is None or not events:
        return {"bin_minutes": 0, "bins": []}
    span = max((end - start).total_seconds(), 60.0)
    width = max(5.0, span / 24)
    n = int(span // width) + 1
    bins: list[dict[str, Any]] = [
        {"start": (start + timedelta(seconds=i * width)).isoformat(), **dict.fromkeys(REAL, 0)}
        for i in range(n)
    ]
    for e in events:
        k = int((datetime.fromisoformat(e["start_time"]) - start).total_seconds() // width)
        bins[min(max(k, 0), n - 1)][e["label"]] += 1
    return {"bin_minutes": round(width / 60, 2), "bins": bins}
