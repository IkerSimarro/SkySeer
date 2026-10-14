"""Build the classifier's labelled track datasets from realistic synthetic scenes.

    python tools/build_training_set.py [--train 400] [--test 150] [--seconds 20]

Writes, in data/datasets/:
- synthetic_train.csv: scenes with seeds 0..train-1 (for training)
- synthetic_test.csv:  seeds 100000.. (held out: never used for training or tuning)
- standard_test.csv:   the 20 clean benchmark scenes (data/bench, if present)

Each scene is rendered, analysed, labelled from its truth and deleted; its rows are
cached in data/datasets/scenes/, so an interrupted build resumes where it stopped.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from skyseer.classify.dataset import (
    build_synthetic,
    object_scores,
    synthetic_rows,
    write_csv,
)
from skyseer.config import load_config

OUT = Path("data/datasets")
TEST_SEED0 = 100_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=int, default=400, help="training scenes")
    ap.add_argument("--test", type=int, default=150, help="held-out test scenes")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    config = load_config()
    started = time.perf_counter()

    def progress(total: int):
        def report(i: int) -> None:
            if (i + 1) % 10 == 0 or i + 1 == total:
                elapsed = time.perf_counter() - started
                print(f"  {i + 1}/{total} scenes ({elapsed / 60:.1f} min)", flush=True)

        return report

    for name, seeds in (
        ("synthetic_train", range(args.train)),
        ("synthetic_test", range(TEST_SEED0, TEST_SEED0 + args.test)),
    ):
        print(f"{name}: {len(seeds)} scenes", flush=True)
        rows = build_synthetic(
            seeds, config, seconds=args.seconds, workers=args.workers, on_scene=progress(len(seeds))
        )
        write_csv(rows, OUT / f"{name}.csv")
        labels = Counter(r["label"] for r in rows if not r["star"])
        kinds = Counter(r["kind"] for r in rows if r["label"] == "Junk")
        rules = object_scores(rows, np.array([r["rule_label"] for r in rows]))
        print(f"  {len(rows)} tracks; labels {dict(labels)}; junk kinds {dict(kinds)}")
        print(f"  rules on these scenes: {rules}", flush=True)

    bench = sorted(Path("data/bench").glob("scene_*_20s.mp4"))
    if bench:
        rows = []
        for video in bench:
            rows += synthetic_rows(video, video.with_suffix(".json"), config, video.stem)
        write_csv(rows, OUT / "standard_test.csv")
        rules = object_scores(rows, np.array([r["rule_label"] for r in rows]))
        print(f"standard_test: {len(rows)} tracks; rules: {rules}")
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
