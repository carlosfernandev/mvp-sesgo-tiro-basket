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
- T2 backend restructure: delegated writer (historical; done in `c698fda`).
- T3 infra/frontend/docs: **inline, one phase at a time** (user directive 2026-10-16:
  no subagents; see Log L1).
- T4 verification: parent-run commands inline.
- T5 work-unit commits: inline.

## Acceptance criteria
1. `.venv/bin/python -m unittest discover -s backend/tests -t backend -p "test_*.py"` -> 84 tests, OK.
2. No root-level `.mp4/.webm/.csv/.png` remain; `data/raw/`, `data/processed/` populated.
3. Frontend upload panel (side/feedback) unchanged, talks to backend via Vite proxy.
4. `docker compose up --build` brings up backend :5001 and frontend :5173.
5. Running the CLI or API from any CWD resolves model/data paths correctly.

## Checklist
- [x] T1. Preserve prior uncommitted work as its own commit (`a031737`).
- [x] T2. Backend restructure + path decoupling + test relocation; 84 tests green.
- [ ] T3. Frontend migration + data cleanup + Docker/infra + README + root package.json
      — decomposed into F1-F4 below.
- [ ] T4. Full verification: tests, frontend build, `docker compose config` + build, smoke
      — executed as F5.
- [ ] T5. Work-unit commits per completed task + RDD assess per commit — executed as F6.

## Phases (T3 decomposed, all inline)
- [x] F1. Frontend move: `git mv packages/web frontend` (20 renames); proxy target from
      `API_PROXY_TARGET` (full URL) with `API_PORT` kept as shorthand, plus `host: true`.
      Check PASS: `npm run build` (`tsc --noEmit && vite build`) exit 0; dev server prints
      `Network: http://192.168.10.114:5288` so `host: true` is effective; decisive proxy
      probe: a control listener on `127.0.0.1:55998` received `GET /api/probe.txt` through
      Vite with `API_PROXY_TARGET=http://127.0.0.1:55998`, proving the env target is honoured
      and not the `5001` default.
- [x] F2. Data cleanup: 9 videos -> `data/raw/`; 3 CSVs + 3 PNGs -> `data/processed/`; the
      legacy `outputs/` results dir merged into `data/processed/` (its `tiro_*`/`tiro_demo`
      session dirs are exactly what the API writes into `results_dir`, so nesting them under
      an `outputs/` subdir would have made the API read `outputs` as a session id).
      Root media files were untracked, so this was a plain `mv`.
      Check PASS: no root `*.mp4|*.webm|*.csv|*.png` left; `git check-ignore` confirms
      `data/raw/*` and `data/processed/*` ignore the payloads (nested session files included)
      while the two `.gitkeep` files stay tracked - only those 2 files entered the index.
      No code path referenced `outputs/`.
- [x] F3. Infra: `backend/Dockerfile` (python:3.11-slim + ffmpeg/libgl1/libglib2.0-0,
      :5001, `FLASK_HOST=0.0.0.0`, path env vars pointed at the mounted volumes so no
      CWD-anchored fallback is ever used), `frontend/Dockerfile` (node:20-alpine, Vite
      dev + HMR, :5173), root `docker-compose.yml` (backend + frontend,
      `API_PROXY_TARGET=http://backend:5001`, bind mounts `./data/raw`, `./data/processed`,
      `./backend/models`), plus one `.dockerignore` per build context.
      Check PASS: `docker compose config` exit 0; the parsed output resolves all three
      bind mounts and the `/app/node_modules` anonymous volume, and frontend
      `depends_on` backend. Image build/runtime smoke is deferred to F5.
- [ ] F4. Docs + root cleanup: README rewrite, root `package.json` (it still points at
      the deleted `packages/api` and at a root-level `test_analisis_tiro`), delete the now
      empty `packages/` and the leftover root `__pycache__/`. Stale references found while
      doing F2, all message/doc text rather than code paths, so they belong to this pass:
      `frontend/README.md` (`packages/api/app.py`, `ANALISIS_RESULTS_DIR=outputs`),
      `frontend/src/components/EmptyState.tsx:13` (`--salida-dir outputs/mi_tiro/` hint shown
      in the UI), and a stale `outputs/` wording in the `app.py` session-id comment.
- [ ] F5. Verification: 84 tests, `tsc --noEmit && vite build`, `docker compose config`.
- [ ] F6. Work-unit commits (branch `multiple-video-detector`).

## Progress
- Baseline before migration: 71 core + 13 API tests OK (Python 3.11.16).
- Branch: multiple-video-detector; RDD: on (default).
- T2 done (gate PASS, parent spot check): `Ran 84 tests / OK` after relocation;
  CWD smoke from /tmp resolves absolute `backend/models/...` + `data/processed`;
  grep gate for `os.getcwd()/os.path.join("models"/sys.path root hack` in backend: empty.
  API `csv` response field now results-dir-relative (verified: no frontend consumer).

## Log
- L1 (2026-10-16, user directive, verbatim):
  "cual es el estado actual? por qué se agota la cuota de peticiones? mejor no utilices
  los agentes y emplea una implementacion por fases"
  Effect: stop launching review subagents; implement T3 in phases inline. The frozen RDD
  review of T2 stays parked (0/4 reviewer results, generation 1, no authority burned).
- L2 (2026-10-16, evidence): root media files are untracked/gitignored, not committed;
  `data/raw` and `data/processed` exist but are empty and have no `.gitkeep`;
  root `package.json` scripts reference the already-deleted `packages/api`.
- L3 (2026-10-16, evidence): OpenCode restarted at 11:38:21 (PID 206978), i.e. after the
  config edit that moved `review-*` and `gentle-orchestrator` to `opencode/big-pickle`
  (6 matches on disk), so the Google free-tier quota is no longer the reason reviewers
  are idle - they   are idle because the user asked not to use them.
- L4 (2026-10-16, user decision via structured question): the 5 `review-*` agents move
  from `opencode/big-pickle` to `opencode/deepseek-v4.1-flash-free` (orchestrator stays on
  big-pickle) so the parked T2 review can be completed with family diversity. Applied to
  `~/.config/opencode/opencode.jsonc` (backup
  `opencode.jsonc.pre-deepseek-review-20261016-114552`); verified: JSONC parses, 14 agents,
  exactly those 5 on deepseek, model present in the catalog. OpenCode does not reload config
  in-process, so an operator restart is required before relaunching the reviewers.
- L6 (2026-10-16, F3 done): wrote `backend/Dockerfile`, `frontend/Dockerfile`,
  `docker-compose.yml` and one `.dockerignore` per build context. `docker compose
  config` parses (exit 0) and resolves all three bind mounts plus the
  `/app/node_modules` anonymous volume. The backend image deliberately does NOT bake
  model weights or media: `MODEL_PATH`, `DATA_RAW_DIR`, `DATA_PROCESSED_DIR` and
  `ANALISIS_RESULTS_DIR` are set to the mounted container paths so the code never
  falls back to the CWD-anchored defaults. Full image build + runtime smoke is
  deferred to F5 (`docker compose up --build`).
- L5 (2026-10-16, evidence): F1 done. The root `node_modules/@sesgo/web` workspace symlink
  pointed at the moved directory, so it was removed (root `node_modules` is gitignored).
  Gotcha for later: `pkill -f 'vite --port <n>'` self-terminates the invoking shell, because
  the pattern also matches that shell's own command line - kill by captured PID instead.
  The dev servers that were listening on 5173/5001 are down (they died with the 11:38:21
  OpenCode restart), which is why the first proxy probe with `http_code=500` proved nothing.

## Next step
1. F4 - docs + root cleanup: README rewrite (via the `cognitive-doc-design` skill), root
   `package.json` (still points at the deleted `packages/api` and at a root-level
   `test_analisis_tiro`), delete the now-empty `packages/` and the leftover root
   `__pycache__/`, and fix the stale `outputs/` wording in `frontend/README.md`,
   `frontend/src/components/EmptyState.tsx:13` and the `app.py` session-id comment.
2. F5 verification: 84 tests, `tsc --noEmit && vite build`, `docker compose config`.
3. F6 work-unit commits per completed phase on `multiple-video-detector` (F3 already
   committed).
4. The frozen RDD review of the T2 candidate stays parked at 3/4 lenses; the
   `review-readability` capture is blocked pending an operator decision (see
   `estado_actual.md`).
