"""Score the rules and the learned model against hand labels of a real recording.

    python tools/score_labels.py data/runs/NAME LABELS.csv [LABELS2.csv ...] [--model PATH]
        [--labelled-run data/runs/OLD]

LABELS.csv has ``event_id,label`` lines (Satellite, Meteor, Plane, Junk or Unsure; Unsure
is ignored). The run must have been analysed with ``--include-junk`` so every track has
an event. Label blind (without looking at SkySeer's answer), and include a random sample
of all events, not only the interesting ones, or the junk rejection can't be measured.

Labels made on an earlier analysis of the same video still work after a code change:
with ``--labelled-run`` each labelled track is matched to the new run's event that
follows it (at least half its detections within 5 px in the same frames). A labelled
track with no match counts as not reported (Junk).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from skyseer.classify.model import LearnedClassifier
from skyseer.config import load_config


def match(labelled: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The event in ``events`` that follows the same object as ``labelled``, if any."""
    want = {d["frame"]: (d["x"], d["y"]) for d in labelled["track"]}
    best, best_n = None, 0
    for e in events:
        if e["last_frame"] < labelled["first_frame"] or e["first_frame"] > labelled["last_frame"]:
            continue
        n = 0
        for d in e["track"]:
            xy = want.get(d["frame"])
            n += xy is not None and (xy[0] - d["x"]) ** 2 + (xy[1] - d["y"]) ** 2 < 25
        if n > best_n:
            best, best_n = e, n
    return best if best_n >= 0.5 * len(want) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("labels", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, default=Path("data/models/classifier.joblib"))
    ap.add_argument("--labelled-run", type=Path, help="the run the labels were made on")
    args = ap.parse_args()
    run_events = json.loads((args.run / "events.json").read_text())
    events: dict[int, dict[str, Any] | None] = {e["id"]: e for e in run_events}
    if args.labelled_run is not None:
        old = json.loads((args.labelled_run / "events.json").read_text())
        events = {e["id"]: match(e, run_events) for e in old}
    labels: dict[int, str] = {}
    for f in args.labels:
        with open(f, encoding="utf-8") as fh:
            labels.update({int(r["event_id"]): r["label"] for r in csv.DictReader(fh)})
    sure = {i: lab for i, lab in labels.items() if lab != "Unsure" and i in events}
    print(f"{len(sure)} labelled events: {dict(Counter(sure.values()))}")

    missing = sum(events[i] is None for i in sure)
    if missing:
        print(f"{missing} labelled tracks have no matching event (counted as not reported)")
    answers = {"rules": {i: (e or {}).get("label", "Junk") for i, e in _of(events, sure)}}
    if args.model.exists():
        model = LearnedClassifier.load(args.model, load_config().rules)
        answers["model"] = {
            i: "Junk" if e is None else str(model.classify(e["features"]).label)
            for i, e in _of(events, sure)
        }
    real = [i for i in sure if sure[i] != "Junk"]
    for name, said in answers.items():
        claims = [i for i in sure if said[i] != "Junk"]
        right = sum(said[i] == sure[i] for i in claims)
        found = sum(said[i] == sure[i] for i in real)
        junk_claimed = sum(sure[i] == "Junk" for i in claims)
        print(
            f"{name:6s} real objects labelled right {found}/{len(real)}; "
            f"claims correct {right}/{len(claims)}; junk called real {junk_claimed}"
        )
        for cls in ("Satellite", "Meteor", "Plane"):
            n = sum(sure[i] == cls for i in sure)
            if n:
                ok = sum(sure[i] == cls and said[i] == cls for i in sure)
                print(f"         {cls}: {ok}/{n}")
        errors = Counter((sure[i], said[i]) for i in sure if said[i] != sure[i])
        print(f"         errors (truth, said): {dict(errors)}")


def _of(events: dict[int, dict[str, Any] | None], ids: dict[int, str]) -> list[tuple[int, Any]]:
    return [(i, events[i]) for i in ids]


if __name__ == "__main__":
    main()
