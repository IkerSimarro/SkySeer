---
name: portfolio-update
description: Update SkySeer's recruiter-facing material after a phase or milestone - README results and features, CHANGELOG, project brief, and showcase images - using only measured, verifiable claims. Use this when a phase is finished, before pushing to GitHub, when the user asks to "update the README", "make the repo look good", "prepare for recruiters", "write release notes", or wants the GitHub page to reflect recent work.
---

# Update the portfolio material

SkySeer exists to get its author a job in the aerospace/space industry, and the GitHub
page is what recruiters and engineers see first. It must be impressive **and** accurate.
An engineer who checks a number and finds it wrong stops trusting everything else.

## 1. Gather facts (don't write from memory)
- What changed: `git log --oneline <last tag or last README update>..HEAD`
- Current numbers: run the `validate` skill (or read `tools/baseline.json` if it ran just
  now) plus the measured results stated in tests and commit messages.
- Test count: `.venv/Scripts/python.exe -m pytest --collect-only -q | tail -1`
- The roadmap and phase status in `docs/PROJECT_BRIEF.md`.

## 2. Update these files
- **`README.md`**, structured for a skimming recruiter:
  1. one-sentence pitch and a hero image or GIF
  2. "What it can do" (capabilities with their measured results)
  3. how it works (pipeline diagram or short list)
  4. results table, with how each number was measured
  5. quick start (`uv sync`, `skyseer analyze ...`)
  6. roadmap
  7. honest limitations
  8. credits and citations (ESA tetra3, Hipparcos, CelesTrak, IAU MDC)
  9. author and links

  Keep the author's existing visual style. Remove claims that no longer match the code;
  the original README had several (e.g. "K-Means separates two classes",
  "varThreshold 45", files that don't exist).
- **`CHANGELOG.md`** (create it if missing): a dated entry per phase or release, in
  plain English.
- **`docs/PROJECT_BRIEF.md`**: phase table and results.
- **Images** in `docs/images/`: calibration overlays, trail galleries, annotated
  frames. Generate them with the footage tools or `skyseer calibrate`. Never commit
  videos or `data/`.

## 3. Recruiter-lens checklist
- Does the first screen show something visual and impressive within 5 seconds?
- Does every number say how it was measured (simulation, real data, test)?
- Are space-industry techniques named, with a short "why it matters" (star tracker /
  plate solving, SGP4 orbit propagation, space situational awareness, photometry)?
- Is the limitations section honest and specific?
- Do the quick-start commands actually work? (Run them.)
- Are third-party data and code credited with licences and citations?

## 4. Finish
Show the user a summary of the changes (or the rendered README) before committing.
Commit locally with a clear message. **Never push to GitHub without the user's explicit
approval**, because the repository is public.
