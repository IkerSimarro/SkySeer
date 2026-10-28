"""One command to check that SkySeer still works and hasn't lost accuracy.

Runs lint, formatting, type checks, the test suite and the synthetic benchmark, then
compares the benchmark with the last accepted numbers in tools/baseline.json.

    python tools/validate.py                  # full check
    python tools/validate.py --quick          # skip the slow test suite and benchmark
    python tools/validate.py --update-baseline  # accept the current benchmark numbers

Exit code 0 = all good, 1 = something failed or accuracy regressed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "baseline.json"
PY = sys.executable

# How much worse than the baseline still counts as "the same" (noise in timing etc.).
TOLERANCE = {"recall": 0.0, "accuracy": 0.0, "false_positives": 0, "frames_per_second": 0.25}


def step(name: str, cmd: list[str]) -> bool:
    t = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    ok = proc.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({time.perf_counter() - t:.0f}s)")
    if not ok:
        # Test output can hold characters a Windows console can't show: replace them
        # rather than crash before the failure is shown.
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
        print((proc.stdout + proc.stderr).strip()[-3000:])
    return ok


def benchmark(scenes: int) -> dict[str, float | int]:
    sys.path.insert(0, str(ROOT / "src"))
    from skyseer.config import load_config
    from skyseer.evaluate import run_benchmark

    return run_benchmark(load_config(), scenes=scenes, cache=ROOT / "data" / "bench").to_dict()


def compare(current: dict, baseline: dict) -> list[str]:
    problems = []
    for key in ("recall", "accuracy"):
        if current[key] < baseline[key] - TOLERANCE[key]:
            problems.append(f"{key} dropped {baseline[key]:.1%} -> {current[key]:.1%}")
    if current["false_positives"] > baseline["false_positives"] + TOLERANCE["false_positives"]:
        problems.append(
            f"false positives rose {baseline['false_positives']} -> {current['false_positives']}"
        )
    fps_floor = baseline["frames_per_second"] * (1 - TOLERANCE["frames_per_second"])
    if current["frames_per_second"] < fps_floor:
        problems.append(
            f"speed dropped {baseline['frames_per_second']} -> "
            f"{current['frames_per_second']} frames/s (machine may be busy)"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="lint + types only")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="save the current benchmark as the new baseline",
    )
    ap.add_argument("--scenes", type=int, default=20)
    args = ap.parse_args()

    ok = all(
        [
            step("ruff lint", [PY, "-m", "ruff", "check", "src", "tests", "tools"]),
            step("ruff format", [PY, "-m", "ruff", "format", "--check", "src", "tests", "tools"]),
            step("mypy types", [PY, "-m", "mypy"]),
        ]
    )
    if args.quick:
        return 0 if ok else 1
    ok &= step("pytest", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"])

    t = time.perf_counter()
    current = benchmark(args.scenes)
    print(
        f"BENCH {args.scenes} scenes ({time.perf_counter() - t:.0f}s): "
        f"recall {current['recall']:.1%}, accuracy {current['accuracy']:.1%}, "
        f"false positives {current['false_positives']}, "
        f"{current['frames_per_second']} frames/s"
    )

    if args.update_baseline or not BASELINE.exists():
        text = json.dumps({"scenes": args.scenes, **current}, indent=2) + "\n"
        BASELINE.write_text(text, newline="\n")  # the same file on Windows and Linux
        print(f"Baseline saved to {BASELINE.relative_to(ROOT)}")
    else:
        baseline = json.loads(BASELINE.read_text())
        if baseline.get("scenes") != args.scenes:
            print(f"NOTE  baseline used {baseline.get('scenes')} scenes; not comparable")
        else:
            problems = compare(current, baseline)
            for p in problems:
                print(f"REGRESSION  {p}")
            # Speed alone (busy machine) is reported but doesn't fail the check.
            ok &= not [p for p in problems if not p.startswith("speed")]
            if not problems:
                print("PASS  benchmark matches or beats the baseline")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
