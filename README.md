# MVP Sesgo Tiro Basket

Extracts sagittal-plane joint angles (elbow, knee, shoulder) from a
pre-recorded basketball shooting video and segments each shot into phases
(`preparacion`, `elevacion`, `liberacion`, `seguimiento`).

The repository is split into two decoupled applications:

| Directory  | What it is                                          | Runtime     |
| ---------- | --------------------------------------------------- | ----------- |
| `backend/` | Analysis engine (`src/core`) + Flask API (`src/api`) | Python 3.11 |
| `frontend/`| React + TypeScript single-page viewer and uploader   | Node 20     |

## Quick path

1. **Backend environment** (Python 3.11):

   ```bash
   python3.11 -m venv .venv
   .venv/bin/pip install -r backend/requirements.txt
   ```

   `ffmpeg` and `ffprobe` must be on `PATH` (used for in-RAM video decoding
   and encoding without writing temporary video files to disk).

2. **Frontend dependencies** — `frontend/` is a standalone Node project:

   ```bash
   cd frontend && npm install
   ```

3. **Run the platform** — from the repository root, in two terminals:

   ```bash
   npm run dev:api   # Flask API on :5001
   npm run dev:web   # Vite dev server on :5173
   ```

   Open <http://localhost:5173>.

4. **Or run everything in containers:**

   ```bash
   docker compose up --build   # API :5001 + frontend :5173
   ```

## Details

| Topic            | Decision                                                                                     |
| ---------------- | -------------------------------------------------------------------------------------------- |
| Interpreter      | Python **3.11**, not 3.14: `mediapipe 1.0.1` ships no 3.14 wheels and cannot install there.   |
| OpenCV           | `backend/requirements.txt` lists `opencv-contrib-python`, NOT `opencv-python`; both break `cv2`. |
| Pose model       | `backend/models/pose_landmarker_lite.task` must be present (mounted at runtime, never baked in). |
| Data layout      | Raw videos in `data/raw/`, results in `data/processed/` (both gitignored, `.gitkeep` only).   |
| Path resolution  | No module resolves paths from the CWD; env vars with absolute `Path(__file__)` fallbacks only. |
| Frontend deps    | `frontend/` owns its `package.json` and `package-lock.json`; install with `npm install`.      |

### Environment variables

| Variable                                  | Used by     | Default                                         |
| ----------------------------------------- | ----------- | ----------------------------------------------- |
| `MODEL_PATH`                              | core        | `backend/models/pose_landmarker_lite.task`      |
| `DATA_RAW_DIR`                            | core        | `data/raw`                                      |
| `DATA_PROCESSED_DIR`                      | core        | `data/processed`                                |
| `ANALISIS_RESULTS_DIR`                    | Flask API   | `DATA_PROCESSED_DIR`, then `data/processed`     |
| `FLASK_HOST` / `FLASK_PORT`               | Flask API   | `127.0.0.1` / `5001`                            |
| `API_PROXY_TARGET` / `API_PORT`           | Vite proxy  | `http://127.0.0.1:5001`                         |

### Command line

```bash
.venv/bin/python backend/src/core/analisis_tiro.py shot.mp4 \
    [--csv angulos_tiro.csv] [--plot perfil_angular.png] \
    [--salida-dir data/processed/mi_tiro/] \
    [--side {auto,left,right}] [--video-feedback feedback.mp4] [--codec mp4v]
```

Outputs:

- `angulos_tiro.csv` (UTF-8): `frame,timestamp,ángulo_codo,ángulo_rodilla,ángulo_hombro,fase_asignada`.
- `perfil_angular.png`: the three angle signals vs time with phase boundaries.
- Console summary: total frames, frames with detection, % missing
  (loud warning if > 10%), min/max per joint.
- `--video-feedback feedback.mp4` (optional): a playable copy of the input
  with the pose skeleton, live joint angles and color-coded phases burned
  in, so the phase segmentation can be validated visually. It reuses the
  exact same angle math and the same `detectar_fases()` call as the CSV, so
  the overlay and the CSV never disagree (single inference pass; coordinates
  are cached, never re-inferred).

Use a distinct `--salida-dir` per shot: without it, every run writes the same
fixed filenames into `DATA_PROCESSED_DIR` and overwrites the previous one.

## Tests and build

```bash
npm test           # both Python suites (core + API)
npm run test:analisis
npm run test:api
npm run build:web  # frontend typecheck + production build
```

## Known limitations

- 2D sagittal-plane approximation: angles are computed on image-plane (pixel)
  projections, not true 3D joint angles. Out-of-plane motion and perspective
  distort them; the clip should be filmed as laterally as possible.
- Side auto-selection heuristic: in a lateral view MediaPipe's LEFT/RIGHT
  labels are anatomical, so the script picks the side with higher mean
  landmark visibility. Override with `--side left|right` when it guesses wrong.
- Phase model assumes a set shot: one elbow dip (minimum), a rise through
  ~90 degrees, a single extension maximum. Clips without that shape produce
  explicit stderr WARNINGs and `sin_deteccion` labels instead of forced phases.
- Frames with no pose detection yield NaN angles (never silently
  interpolated) and are labeled `sin_deteccion`.
- Video codec: the feedback MP4 defaults to `mp4v` (MPEG-4 Part 2 in an MP4
  container), which VLC and desktop players play fine but some browsers do
  not; `avc1` (H.264) is the better choice on machines whose encoder
  supports it — pass `--codec avc1`. If the encoder is unavailable the
  script raises an explicit error instead of writing a 0-byte file.
