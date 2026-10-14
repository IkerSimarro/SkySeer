"""Labelled tracks for training and testing the classifier.

One row per track: every measurement (``features.FEATURE_NAMES``, so the rules' opinion
can be recomputed if the rules change), the rules' opinion (``model.RULE_FEATURES``),
the true label, and the video it came from (its *group*, so one video's tracks never
end up on both sides of a train/test split).

Where labels come from:

- **Synthetic scenes** with exact truth (``synthetic_rows``): a track takes the class of
  the truth object it follows, or Junk when it follows nothing real (noise, insects,
  twinkling stars, clouds...).
- **Real recordings** (``run_rows``): events matched to a satellite's orbit are labelled
  Satellite by orbital mechanics, with no human needed, plus any events labelled by
  hand in a labels file (``event_id,label`` per line).
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..features import FEATURE_NAMES
from ..models import ObjectClass
from .model import CLASSES, MODEL_FEATURES, RULE_FEATURES, rule_shares
from .rules import RuleClassifier

log = logging.getLogger(__name__)

# truth_id: which real object (index in the scene's truth) the track follows, -1 if none;
# real_objects: how many real objects the scene had (detected or not); star: the
# star-field filter claimed the track. These allow object-level scoring of any classifier
# from the rows alone (detection and tracking don't depend on the classifier).
META_COLUMNS = (
    "label",
    "kind",
    "group",
    "event_id",
    "rule_label",
    "truth_id",
    "real_objects",
    "star",
)
COLUMNS = FEATURE_NAMES + RULE_FEATURES + META_COLUMNS


def _row(features: dict[str, float], rules: RuleClassifier, **meta: Any) -> dict[str, Any]:
    shares = rule_shares(features, rules)
    rule_label = CLASSES[int(np.argmax([shares[name] for name in RULE_FEATURES]))]
    return {**features, **shares, "rule_label": rule_label, **meta}


def synthetic_rows(
    video: str | Path, truth_path: str | Path, config: Config, group: str
) -> list[dict[str, Any]]:
    """Run the pipeline on a synthetic video and label every track from its truth."""
    from ..evaluate import _match_score
    from ..pipeline import run
    from ..sources import FileSource
    from ..synth import GroundTruth, TruthObject

    truth = GroundTruth.load(Path(truth_path))
    real = [o.label != str(ObjectClass.JUNK) for o in truth.objects]
    rules = RuleClassifier(config.rules)
    rows = []
    for event in run(FileSource(video), config):
        best, score = -1, 0.5
        for k, candidate in enumerate(truth.objects):
            s = _match_score(event, candidate, 15.0)
            if s >= score:
                best, score = k, s
        obj: TruthObject | None = truth.objects[best] if best >= 0 else None
        rows.append(
            _row(
                event.features,
                rules,
                label=obj.label if obj is not None else str(ObjectClass.JUNK),
                kind=(obj.kind or "object") if obj is not None else "noise",
                group=group,
                event_id=event.id,
                truth_id=best if best >= 0 and real[best] else -1,
                real_objects=sum(real),
                star=int(event.label is ObjectClass.STAR),
            )
        )
    return rows


def run_rows(
    events_json: str | Path,
    config: Config,
    labels_csv: str | Path | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    """Rows from a real recording's events.json: orbit-matched satellites, plus hand labels.

    Only events whose label is known are returned; the rest are unlabelled.
    """
    events = json.loads(Path(events_json).read_text(encoding="utf-8"))
    hand: dict[int, str] = {}
    if labels_csv is not None:
        with open(labels_csv, encoding="utf-8") as fh:
            for line in csv.DictReader(fh):
                hand[int(line["event_id"])] = line["label"].strip().capitalize()
    rules = RuleClassifier(config.rules)
    group = group or Path(events_json).parent.name
    rows = []
    for e in events:
        if e["id"] in hand:
            label, kind = hand[e["id"]], "hand"
        elif e.get("label_source") == "orbit":
            label, kind = str(ObjectClass.SATELLITE), "orbit"
        else:
            continue
        if label not in CLASSES:
            continue
        rows.append(
            _row(e["features"], rules, label=label, kind=kind, group=group, event_id=e["id"])
        )
    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in COLUMNS})
    return path


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X features, y labels, groups, rules' labels) for scikit-learn."""

    def num(v: Any) -> float:
        return float(v) if v not in (None, "") else np.nan

    X = np.array([[num(r[name]) for name in MODEL_FEATURES] for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    rule_labels = np.array([r["rule_label"] for r in rows])
    return X, y, groups, rule_labels


def object_scores(rows: list[dict[str, Any]], predicted: np.ndarray) -> dict[str, float]:
    """Benchmark-style scores from rows and a classifier's labels (same rules as
    ``evaluate.evaluate``): recall and accuracy over real objects, and false positives
    (tracks following no real object that were not called Junk or Star)."""
    labels = ["Star" if int(r["star"]) else str(p) for r, p in zip(rows, predicted, strict=True)]
    best: dict[tuple[str, int], tuple[int, str, str]] = {}
    scenes: dict[str, int] = {}
    false_positives = 0
    for r, label in zip(rows, labels, strict=True):
        scenes[r["group"]] = int(r["real_objects"])
        tid = int(r["truth_id"])
        if tid < 0:
            false_positives += label not in ("Junk", "Star")
            continue
        n = int(float(r["detection_count"]))
        key = (r["group"], tid)
        if key not in best or n > best[key][0]:  # judged by its longest piece
            best[key] = (n, label, r["label"])
    total = sum(scenes.values())
    correct = sum(1 for _, label, true in best.values() if label == true)
    return {
        "objects": total,
        "recall": round(len(best) / total, 4) if total else 1.0,
        "accuracy": round(correct / len(best), 4) if best else 0.0,
        "false_positives": false_positives,
    }


def _scene_rows(args: tuple[int, float, str, Config]) -> list[dict[str, Any]]:
    """Render one realistic scene, label its tracks and delete the video (it can be
    re-rendered exactly from its seed). The rows are cached, so builds can resume."""
    import tempfile

    from ..synth import generate, realistic_spec

    seed, seconds, cache, config = args
    cached = Path(cache) / f"scene_{seed:06d}.json"
    if cached.exists():
        rows: list[dict[str, Any]] = json.loads(cached.read_text(encoding="utf-8"))
        return rows
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "scene.mp4"
        generate(realistic_spec(seed, seconds), video)
        rows = synthetic_rows(video, video.with_suffix(".json"), config, f"synthetic-{seed}")
    cached.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def build_synthetic(
    seeds: Iterable[int],
    config: Config,
    cache: str | Path = "data/datasets/scenes",
    seconds: float = 20.0,
    workers: int | None = None,
    on_scene: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Rows from realistic synthetic scenes, built in parallel and cached per scene."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    Path(cache).mkdir(parents=True, exist_ok=True)
    jobs = [(seed, seconds, str(cache), config) for seed in seeds]
    rows: list[dict[str, Any]] = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, scene_rows in enumerate(pool.map(_scene_rows, jobs)):
            rows.extend(scene_rows)
            if on_scene is not None:
                on_scene(i)
    return rows
