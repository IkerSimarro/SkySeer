---
name: validate
description: Run SkySeer's full quality gate (lint, formatting, type checks, all tests, and the synthetic accuracy benchmark compared with the accepted baseline) and report in plain English whether anything broke or lost accuracy. Use this before every commit, after any change to detection, tracking, classification or astronomy code, when the user asks "does everything still work?", "run the checks", "did I break anything", or before pushing to GitHub, even if they don't say "validate".
---

# Validate SkySeer

## Run it

From the repository root:

```
.venv/Scripts/python.exe tools/validate.py          # full: ~5 minutes
.venv/Scripts/python.exe tools/validate.py --quick  # lint + types only: seconds
```

(On Linux or the Pi use `.venv/bin/python`, or `uv run python tools/validate.py`.) The full
run takes several minutes, so start it in the background and continue with other work.

The script prints PASS or FAIL per step, the benchmark numbers (detection recall,
classification accuracy, false positives, frames per second), and compares them with
`tools/baseline.json`.

## Report to the user

Keep it short and plain:
- **All passed:** one line with the benchmark numbers.
- **Something failed:** what failed, the relevant error lines, and the likely cause.
  Then fix it if it's clearly caused by the current change, or ask if not.
- **Accuracy regressed** (lower recall or accuracy, or more false positives): treat it
  as a real bug. Investigate which scenes or classes changed (`skyseer bench` prints the
  confusion matrix) rather than adjusting thresholds until the number comes back.
- **Only speed regressed:** often just a busy machine, so mention it without alarm and
  re-run if it matters.

## The baseline

`tools/baseline.json` holds the last *accepted* numbers. Update it
(`tools/validate.py --update-baseline`) only when a change genuinely improves results, or
deliberately trades one metric for another, **and the user agrees**. Never update it to
make a regression disappear: the point of the baseline is that accuracy claims in the
README stay true.

## Before committing

Run the full validation, then commit with a message that states measured effects
("benchmark unchanged: 100% / 100% / 0 FP"). Never push to GitHub without the user's
explicit OK.
