# frontend — shot-analysis viewer

Single-page React + TypeScript app (Vite, React 18, Recharts) that visualizes
shot-analysis sessions and lets users upload and analyze shooting videos
in memory via the backend API (`backend/src/api/app.py`).

## Development

The frontend is a **standalone** Node project with its own dependencies, so
install them once inside `frontend/`:

```bash
cd frontend && npm install
```

Then start both services from the repository root in separate terminals:

```bash
# Terminal 1 — API (from the repo root):
npm run dev:api
# Or directly with Python:
.venv/bin/python backend/src/api/app.py

# Terminal 2 — Frontend (from the repo root):
npm run dev:web
# Or from frontend/:
cd frontend && npm run dev
```

The frontend runs on `http://localhost:5173`. The Vite dev proxy
(`vite.config.ts`) forwards same-origin `/api/*` calls to
`http://127.0.0.1:5001` by default; set `API_PROXY_TARGET` to point at a
different host/port (for example `http://backend:5001` inside Docker Compose),
or `API_PORT` as a shorthand for the local port. No component hardcodes a host
or port — see `src/api.ts`.

## Structure

- `src/api.ts` — centralized API config + fetch helpers (same-origin paths).
- `src/types.ts` — API contract types. Accented CSV keys (`ángulo_codo`,
  …) are kept verbatim; `toChartPoints` maps them once, explicitly, to ASCII
  chart keys so a typo fails at that boundary instead of silently charting
  `undefined`.
- `src/components/SessionPicker.tsx` — session list with frame count + fps.
- `src/components/EmptyState.tsx` — first-class `[]` state with the
  `--salida-dir data/processed/<nombre>/` guidance.
- `src/components/VideoPlayer.tsx` — video + frame readout from
  `currentTime * fps`.
- `src/components/PhaseTimeline.tsx` — clickable phase strip from
  `phase_spans`, with playhead.
- `src/components/AngleChart.tsx` — Recharts plot of `ángulo_codo` with
  phase background bands and optional rodilla/hombro series. Hovering seeks
  the video; video playback moves the cursor (two-way sync owned by `App`).
