# MVP Sesgo Tiro Basket

Extracts sagittal-plane joint angles (elbow, knee, shoulder) from a
pre-recorded basketball shooting video and segments the shot into phases
(`preparacion`, `elevacion`, `liberacion`, `seguimiento`).

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Why 3.11 and not 3.14: `mediapipe 1.0.1` ships no Python 3.14 wheels, so it
cannot even be installed there. Python 3.11 is a version with available
wheels and is therefore the pinned interpreter for this MVP.

Note: `requirements.txt` lists `opencv-contrib-python`, NOT `opencv-python`.
Installing both breaks `cv2` because they ship the same module.

The pose model (`models/pose_landmarker_lite.task`) must already be present.

## Run

```bash
.venv/bin/python analisis_tiro.py shot.mp4 [--csv angulos_tiro.csv] \
    [--plot perfil_angular.png] [--side {auto,left,right}] \
    [--video-feedback feedback.mp4] [--codec mp4v]
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

Run the tests with:

```bash
.venv/bin/python -m unittest test_analisis_tiro -v
```

## Web Platform (Backend API + Frontend Web)

The repository includes a web platform allowing users to inspect historical runs or upload MP4 videos directly through a graphical interface, configuring dominant side (`--side`) and visual feedback overlay (`--feedback`) via UI controls instead of terminal arguments.

### Prerequisites

1. **Backend dependencies:**
   ```bash
   python3.11 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/pip install -r requirements-api.txt
   ```
   *System requirement:* `ffmpeg` and `ffprobe` must be present on PATH (used for in-RAM decoding and encoding without writing temporary video files to disk).

2. **Frontend dependencies:**
   ```bash
   npm install
   ```

### Running the Platform

From the repository root, start both services in separate terminals:

- **Terminal 1 — Backend API (Flask on port 5001):**
  ```bash
  npm run dev:api
  ```
  *(Runs `packages/api/app.py` bound to port 5001. Override results directory with `ANALISIS_RESULTS_DIR` if needed).*

- **Terminal 2 — Frontend (Vite on port 5173):**
  ```bash
  npm run dev:web
  ```
  *(Runs `packages/web` with Vite dev server. The proxy automatically routes `/api/*` requests to `http://127.0.0.1:5001`).*

Open **http://localhost:5173** in your web browser.

### Running Tests & Build

```bash
# Core analysis tests (Python)
npm run test:analisis

# API tests (Python)
npm run test:api

# Frontend build & typecheck (TypeScript + Vite)
npm run build:web
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
