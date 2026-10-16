# Clean architecture migration: frontend/ + backend/ + Docker

## Objective
Migrate the monorepo layout (`packages/*`, root-level Python and media files) to a
decoupled `frontend/` + `backend/` structure with Docker infrastructure, CV/DS
directory conventions (`data/raw`, `data/processed`), and zero CWD coupling in
Python path resolution.

## Why
- Root is polluted with videos, CSVs, plots, and a 1900-line module.
- `DEFAULT_MODEL_PATH`, CLI output paths, and API results dir resolve relative
  to the process CWD (`os.path.join("models", ...)`, `"angulos_tiro.csv"`).
- No container story; frontend/backend startup is manual and coupled to the
  monorepo workspaces layout.

## Scope
In scope:
- `backend/`: core engine (`src/core`), Flask API (`src/api`), tests, models, consolidated requirements.
- `frontend/`: migration of `packages/web` (React 18 + Vite + TS) with proxy env support.
- Path decoupling: `MODEL_PATH`, `DATA_RAW_DIR`, `DATA_PROCESSED_DIR` env vars with
  absolute `Path(__file__)` fallbacks.
- Data cleanup: root videos -> `data/raw/`, CSVs/plots/session outputs -> `data/processed/`.
- `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, README rewrite.
- All 84 existing tests (71 core + 13 API) relocated and green.

Out of scope (deliberately deferred):
- Any change to analysis/phase/shot-detection logic.
- UI changes beyond proxy/env wiring.
- CI pipelines, reverse proxy, production deployment.
- Known follow-up from prior feature: MAX_SHOT_SPAN_MS false positive (shot-detection T5).

## Constraints
- No Python module may resolve paths relative to CWD; env var with
  `Path(__file__)`-anchored absolute fallback only.
- Media files under `data/` are gitignored (`.gitkeep` only); models stay in
  `backend/models/` and are volume-mounted, not baked into images.
- Backend image: `python:3.11-slim` + `ffmpeg`, `libgl1`, `libglib2.0-0`, port 5001.
- Frontend image: `node:20-alpine`, dev server with HMR, port 5173.
- Existing public contracts preserved: API routes (`/api/analizar`, `/api/sesiones*`),
  `create_app(results_dir=...)` signature, CLI of `analisis_tiro`, upload UI with
  `--side` / `--feedback` selectors.
- TDD: off (refactor under an existing 84-test regression suite; tests are the verification).
- Delivery: work-unit commits on `multiple-video-detector`; no PR unless requested.

## Route declaration
- T1 commit prior WIP: inline (git state only).
- T2 backend restructure: delegated (writer trigger: 2+ non-trivial files).
- T3 infra/frontend/docs: delegated (writer trigger: 2+ non-trivial files).
- T4 verification: parent-run commands + bounded docker build (per-action).

## Acceptance criteria
1. `.venv/bin/python -m unittest discover -s backend/tests -t backend -p "test_*.py"` -> 84 tests, OK.
2. No root-level `.mp4/.webm/.csv/.png` remain; `data/raw/`, `data/processed/` populated.
3. Frontend upload panel (side/feedback) unchanged, talks to backend via Vite proxy.
4. `docker compose up --build` brings up backend :5001 and frontend :5173.
5. Running the CLI or API from any CWD resolves model/data paths correctly.

## Checklist
- [x] T1. Preserve prior uncommitted work as its own commit (`a031737`).
- [x] T2. Backend restructure + path decoupling + test relocation; 84 tests green.
- [ ] T3. Frontend migration + data cleanup + Docker/infra + README + root package.json.
- [ ] T4. Full verification: tests, frontend build, `docker compose config` + build, smoke.
- [ ] T5. Work-unit commits per completed task + RDD assess per commit.

## Progress
- Baseline before migration: 71 core + 13 API tests OK (Python 3.11.16).
- Branch: multiple-video-detector; RDD: on (default).
- T2 done (gate PASS, parent spot check): `Ran 84 tests / OK` after relocation;
  CWD smoke from /tmp resolves absolute `backend/models/...` + `data/processed`;
  grep gate for `os.getcwd()/os.path.join("models"/sys.path root hack` in backend: empty.
  API `csv` response field now results-dir-relative (verified: no frontend consumer).

## Next step
T2 via delegated writer.
