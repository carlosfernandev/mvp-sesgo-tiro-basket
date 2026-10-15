# packages/web — visor de análisis de tiro

Single-page React + TypeScript app (Vite, React 18, Recharts) that visualizes
the shot-analysis sessions served by `packages/api/app.py`. It never talks to
the analysis pipeline directly — only to the read-only Flask API.

## Desarrollo

```bash
# Terminal 1 — API (from the repo root):
ANALISIS_RESULTS_DIR=outputs .venv/bin/python -m flask \
  --app packages/api/app.py run --port 5097

# Terminal 2 — frontend (from packages/web):
npm install
npm run dev     # http://localhost:5173, /api/* proxied to :5097
```

The Vite dev proxy (`vite.config.ts`) forwards same-origin `/api/*` calls to
`http://127.0.0.1:5097` by default; set `API_PORT` to point elsewhere. No
component hardcodes a host or port — see `src/api.ts`.

## Estructura

- `src/api.ts` — centralized API config + fetch helpers (same-origin paths).
- `src/types.ts` — API contract types. Accented CSV keys (`ángulo_codo`,
  …) are kept verbatim; `toChartPoints` maps them once, explicitly, to ASCII
  chart keys so a typo fails at that boundary instead of silently charting
  `undefined`.
- `src/components/SessionPicker.tsx` — session list with frame count + fps.
- `src/components/EmptyState.tsx` — first-class `[]` state with the
  `--salida-dir outputs/<nombre>/` guidance.
- `src/components/VideoPlayer.tsx` — video + frame readout from
  `currentTime * fps`.
- `src/components/PhaseTimeline.tsx` — clickable phase strip from
  `phase_spans`, with playhead.
- `src/components/AngleChart.tsx` — Recharts plot of `ángulo_codo` with
  phase background bands and optional rodilla/hombro series. Hovering seeks
  the video; video playback moves the cursor (two-way sync owned by `App`).
