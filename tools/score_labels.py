"""Score the rules and the learned model against hand labels of a real recording.

    python tools/score_labels.py data/runs/NAME LABELS.csv [LABELS2.csv ...] [--model PATH]

LABELS.csv has ``event_id,label`` lines (Satellite, Meteor, Plane, Junk or Unsure; Unsure
is ignored). The run must have been analysed with ``--include-junk`` so every track has
an event. Label blind (without looking at SkySeer's answer), and include a random sample
of all events, not only the interesting ones, or the junk rejection can't be measured.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from skyseer.classify.model import LearnedClassifier
from skyseer.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("labels", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, default=Path("data/models/classifier.joblib"))
    args = ap.parse_args()
    events = {e["id"]: e for e in json.loads((args.run / "events.json").read_text())}
    labels: dict[int, str] = {}
    for f in args.labels:
        with open(f, encoding="utf-8") as fh:
            labels.update({int(r["event_id"]): r["label"] for r in csv.DictReader(fh)})
    sure = {i: lab for i, lab in labels.items() if lab != "Unsure" and i in events}
    print(f"{len(sure)} labelled events: {dict(Counter(sure.values()))}")

    answers = {"rules": {i: events[i]["label"] for i in sure}}
    if args.model.exists():
        model = LearnedClassifier.load(args.model, load_config().rules)
        answers["model"] = {i: str(model.classify(events[i]["features"]).label) for i in sure}
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


if __name__ == "__main__":
    main()
