# SkySeer: notes for Claude

Night-sky monitoring system: detects, tracks and classifies satellites, meteors and
aircraft in video; calibrates the camera from the stars (plate solving); names
satellites from public orbit data; associates meteors with showers; measures magnitudes.
Full overview, results and roadmap: `docs/PROJECT_BRIEF.md`.

## Why it exists: read this first
It is the author's **portfolio project for getting a job in the aerospace/space
industry**. Judge decisions partly by how an aerospace recruiter or engineer would see
them on GitHub: measured results, honest limitations, recognisable space-industry
techniques, clean tested code.

## Working with the author
- **Explain in plain English.** They're a student and want to understand every decision;
  define technical terms briefly when first used.
- **Plan in phases and report honestly:** what works, what doesn't, why, and the numbers.
- **Claim only what is measured** (tests, benchmark, simulations, real data). Prefer
  refusing over a confident wrong answer (wrong satellite name, false sky position).
- **Never push to GitHub, rewrite history or force-push without explicit approval.** The
  repo is public. Commit locally on the feature branch (`refactor/foundation`).
- Give short progress updates during long tasks.

## Commands (Windows dev machine; run from the repo root)
- `.venv\Scripts\python.exe -m pytest -q`: all tests (~3-4 min; astro tests build a
  small star database in `data/astro` on first run)
- `.venv\Scripts\python.exe tools\validate.py`: lint + format + mypy + tests + benchmark
  against `tools/baseline.json` (use `--quick` for lint/types only)
- `.venv\Scripts\skyseer.exe analyze|calibrate|live|serve|train|synth|bench|orbits|showers ...`: CLI
- Dashboard: `skyseer serve` (http://localhost:8000, API docs at /api/docs). Frontend source in
  `web/` (React + TypeScript + Vite): `cd web; npm run build` compiles it into
  `src/skyseer/api/static/` (git-ignored); `npm run dev` for live reload against a running
  `skyseer serve`. Station database: `data/skyseer.db` (override with `SKYSEER_DATABASE`).
  Live station: `skyseer serve --camera 0`, or without a camera
  `skyseer serve --replay VIDEO --start-time ... --lat ... --lon ...` (plays a recording as
  if live). `.claude/launch.json` has both for the preview browser.
- `uv sync` installs/updates the environment. If `uv` isn't on PATH in a fresh shell,
  refresh PATH from the registry or call the venv's python directly.

## Conventions
- Package in `src/skyseer/` (astronomy in `src/skyseer/astro/`), tests in `tests/`,
  dev tools in `tools/`.
- ruff (line length 100) + mypy must pass; add or extend tests with every behaviour
  change; keep the benchmark at baseline (see the `validate` skill).
- Settings live in `src/skyseer/config.py` **and** `configs/default.yaml`; a test fails if
  they differ. Update both.
- `src/skyseer/astro/_vendor/tetra3.py` is vendored ESA code: only minimal, documented
  patches (listed in its header).
- Bundled data: `src/skyseer/astro/data/showers.json` (rebuild with
  `tools/build_showers.py`). Downloaded data (star catalogue, orbits, databases), videos
  and run outputs live in `data/` and `samples/`, which are git-ignored. Never commit
  videos.
- Commit messages describe *measured* effects; end them with the Co-Authored-By line.

## Environment gotchas (Windows)
- PowerShell 5.1: `Set-Content -Encoding utf8` adds a BOM (it broke `pyproject.toml`
  once); use the file tools or bash. It also strips double quotes from native-command
  arguments, so pass JSON arguments via bash.
- Long analyses of real videos take minutes; run them in the background.

## Project skills (`.claude/skills/`)
`validate` (quality gate), `analyze-footage` (real-video investigation workflow),
`simulate` (synthetic scenarios with known truth), `interview-prep` (explain, quiz and
mock-interview the author), `portfolio-update` (README/CHANGELOG with measured claims).
