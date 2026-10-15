"""HTTP API serving shot-analysis artifacts plus an upload-and-process endpoint.

Read-only endpoints (GET /api/sesiones...) serve already-generated artifacts
exactly as before. POST /api/analizar additionally RUNS the analysis on an
uploaded MP4 entirely in RAM: the video bytes are piped through ffmpeg and
analisis_tiro.analizar_frames_con_cache without ever touching the filesystem,
and ONLY the resulting angulos_tiro.csv is persisted to outputs/<id>/. The
annotated feedback video is re-encoded in memory (fragmented H.264 MP4) and
kept in a process-local store served at GET /api/analizar/<id>/video, so the
frontend can play it immediately; it is ephemeral (lost on restart) while the
CSV is the persistent artifact.

Upload contract (POST /api/analizar): send the MP4 either as the raw request
body with Content-Type: video/mp4, or as multipart/form-data with the file in
the "video" field (alias "file" is also accepted). Uploads larger than 200 MB
are rejected with 413 and never processed. Success (200) returns JSON with
the id, frame_count, fps, width, height, phase_spans, rows, csv path,
video_url and warnings; the video bytes are fetched with a separate GET to
video_url. Every failure returns JSON {"error": ...} with an actionable
message and never a traceback.

Results directory resolution: the ``ANALISIS_RESULTS_DIR`` environment
variable wins when set; otherwise the repo root is used (two levels above
this file: ``packages/api/`` -> monorepo root). The historical default output
of ``analisis_tiro.py`` IS the repo root, so defaulting there keeps local
development working with zero configuration.

A session is an immediate subdirectory containing ``angulos_tiro.csv``. A CSV
sitting directly at the results-dir root is NOT a session: it is the legacy
single-run layout and is ignored, so pointing the variable at a repo root full
of loose artifacts still yields a clean listing.
"""

import csv
import glob
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone

# WHY a hardcoded env-var name instead of a CLI flag: the npm workspace script
# (`npm run dev` in packages/api) and the Vite frontend each spawn the server
# in different ways, and an env var is the one configuration channel both
# share without argument plumbing.
RESULTS_ENV_VAR = "ANALISIS_RESULTS_DIR"

# WHY this exact filename: it is the default `--csv` basename written by
# analisis_tiro.py, so a session is recognized by the file the pipeline
# always produces, not by the optional ones.
CSV_FILENAME = "angulos_tiro.csv"

# WHY a glob instead of a fixed name: the video basename is whatever the caller
# passed to `--video-feedback` (e.g. `feedback.mp4`, `feedback2.mp4`,
# `shot3_feedback.mp4`), and `--salida-dir` defaults it to plain `feedback.mp4`.
# The earlier pattern `*_feedback*.mp4` assumed an infix that generar_video_
# feedback() never actually adds, so it matched none of the real files.
VIDEO_GLOB = "*feedback*.mp4"

# WHY this exact filename: it is the default `--plot` basename written by
# analisis_tiro.py.
PERFIL_FILENAME = "perfil_angular.png"

# WHY this fixed column list: the frontend parses rows positionally-free by
# name, so a pipeline change that renames a column must fail loudly here
# (session marked with an error) instead of silently serving shifted data.
EXPECTED_COLUMNS = (
    "frame",
    "timestamp",
    "ángulo_codo",
    "ángulo_rodilla",
    "ángulo_hombro",
    "fase_asignada",
)

# WHY localhost-only instead of "*": the Vite dev server runs on a different
# port of the same machine, so reflecting any localhost origin is enough for
# development while a wildcard would also let arbitrary websites read session
# data once this URL is pasted into a browser.
LOCALHOST_ORIGIN_RE = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$")

# WHY 200 MB: a phone slow-motion shot clip is tens of MB; anything larger is
# almost certainly a wrong file (a full session recording) that would OOM the
# worker when piped through rawvideo decode. The cap is checked on the
# Content-Length header AND on the actual bytes (chunked uploads have no
# header), and the response is JSON 413, never a traceback or a bare socket
# reset.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# WHY absolute ffmpeg/ffprobe paths instead of bare names: the venv does not
# ship them and a bare name would silently pick whatever is on PATH (or fail
# obscurely); the absolute path fails fast with a clear message when missing.
FFMPEG_BIN = "/usr/bin/ffmpeg"
FFPROBE_BIN = "/usr/bin/ffprobe"

# WHY libopenh264 and not libx264: this ffmpeg build was compiled WITHOUT
# libx264 (check `ffmpeg -encoders | grep 264`: only libopenh264 is listed),
# so requesting libx264 fails at encode time. libopenh264 IS real H.264
# (codec_name=h264 on probe), which is exactly what Chrome/Firefox need; the
# old cv2.VideoWriter default (mp4v = MPEG-4 Part 2, codec_name=mpeg4) is what
# browsers cannot play, which is the past bug this endpoint fixes.
VIDEO_ENCODER = "libopenh264"

# WHY fragmented MP4 flags are mandatory: a normal MP4 muxer writes its moov
# header by SEEKING back to the start of the file, but a stdout pipe is not
# seekable, so without these flags ffmpeg aborts with "muxer does not support
# non seekable output". frag_keyframe+empty_moov+default_base_moof switches
# the muxer to fragmented MP4 (moov up front, one moof per fragment), which
# streams cleanly over a pipe AND gives browsers a faststart file for free.
FRAG_MOVFLAGS = "frag_keyframe+empty_moov+default_base_moof"

# WHY ffmpeg pipes instead of a temp file: this is a privacy requirement, not
# an optimization. The uploaded video must NEVER touch the filesystem (no
# NamedTemporaryFile, no open(wb) of video bytes), so decode reads MP4 bytes
# from ffmpeg stdin (pipe:0) and yields raw BGR frames from stdout (pipe:1)
# one frame at a time. Only the resulting CSV is ever written to disk.
# Streaming frame by frame (never buffering the whole raw clip, which would
# be gigabytes for a 10 s 1080p video) is WHY the decoder yields numpy arrays
# instead of returning a list.

# In-memory store for encoded feedback videos: session id -> MP4 bytes.
# WHY process-local memory and not disk: the privacy rule above forbids
# writing the video anywhere, so the bytes live only here between the POST
# and the client's GET. Ephemeral by design: a restart drops them while the
# CSV (the persistent artifact) survives. Guarded by a lock because Flask
# may serve requests from multiple threads.
_VIDEO_STORE = {}
_VIDEO_LOCK = threading.Lock()

from flask import Flask, Response, jsonify, request, send_file


def create_app(results_dir=None):
    """Build the Flask application bound to one results directory.

    Args:
        results_dir: directory whose immediate subdirectories are scanned for
            sessions. None resolves via ANALISIS_RESULTS_DIR with fallback to
            the repo root, so tests can inject a fixtures directory without
            touching the environment.

    Returns:
        The configured Flask application.
    """
    app = Flask(__name__)
    app.config["RESULTS_DIR"] = _resolve_results_dir(results_dir)

    @app.after_request
    def _cors_localhost(response):
        origin = request.headers.get("Origin", "")
        if LOCALHOST_ORIGIN_RE.match(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            # WHY POST is listed: the upload endpoint needs it for browser
            # CORS preflights; the read-only endpoints only ever needed GET.
            response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS, POST"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Range"
        return response

    @app.get("/api/health")
    def health():
        """Liveness payload; also reports which directory is being served.

        Args: none (query parameters are ignored).

        Returns: JSON object with status ok and the absolute results dir.
        """
        return jsonify({"status": "ok", "results_dir": app.config["RESULTS_DIR"]})

    @app.get("/api/sesiones")
    def list_sesiones():
        """List every session found in the results directory.

        Args: none.

        Returns: JSON array, one summary object per session, empty array with
            HTTP 200 when there are no runs. A session whose CSV cannot be
            parsed is included with an `error` field instead of failing the
            whole listing.
        """
        return jsonify([_summarize_session(app.config["RESULTS_DIR"], sid)
                        for sid in _session_ids(app.config["RESULTS_DIR"])])

    @app.get("/api/sesiones/<sid>")
    def sesion_detail(sid):
        """Full detail of one session: metadata plus every CSV row as JSON.

        Args:
            sid: session id, i.e. the directory name inside the results dir.

        Returns: JSON object with metadata and rows; 400 for a traversal
            attempt, 404 when the session or its CSV does not exist, 422
            when the CSV is present but unparsable.
        """
        error = _reject_traversal(sid)
        if error is not None:
            return error
        results_dir = app.config["RESULTS_DIR"]
        csv_path = os.path.join(results_dir, sid, CSV_FILENAME)
        if not os.path.isdir(os.path.join(results_dir, sid)) or not os.path.isfile(csv_path):
            return jsonify({"error": "session '%s' not found" % sid}), 404
        try:
            rows = _read_rows(csv_path)
        except ValueError as exc:
            return jsonify({"error": "session '%s': %s" % (sid, exc)}), 422
        summary = _summarize_session(results_dir, sid)
        summary["rows"] = rows
        return jsonify(summary)

    @app.get("/api/sesiones/<sid>/video")
    def sesion_video(sid):
        """Stream the session feedback video with HTTP Range support.

        Args:
            sid: session id (directory name inside the results dir).

        Returns: the MP4 bytes; 206 Partial Content when the client sends a
            Range header (which is what lets the browser seek), 400 for a
            traversal attempt, 404 when the session has no video.
        """
        error = _reject_traversal(sid)
        if error is not None:
            return error
        video_path = _find_video(app.config["RESULTS_DIR"], sid)
        if video_path is None:
            return jsonify({"error": "session '%s' has no video" % sid}), 404
        # WHY conditional=True: it is the single flag that makes Werkzeug
        # honor Range requests with 206 Partial Content; without it Flask
        # always returns 200 and the browser <video> element cannot seek,
        # which defeats the entire purpose of this endpoint.
        return send_file(video_path, mimetype="video/mp4", conditional=True)

    @app.get("/api/sesiones/<sid>/perfil")
    def sesion_perfil(sid):
        """Serve the session angular-profile PNG.

        Args:
            sid: session id (directory name inside the results dir).

        Returns: the PNG bytes; 400 for a traversal attempt, 404 when the
            session has no profile image.
        """
        error = _reject_traversal(sid)
        if error is not None:
            return error
        png_path = os.path.join(app.config["RESULTS_DIR"], sid, PERFIL_FILENAME)
        if not os.path.isfile(png_path):
            return jsonify({"error": "session '%s' has no profile image" % sid}), 404
        return send_file(png_path, mimetype="image/png", conditional=True)

    @app.post("/api/analizar")
    def analizar_upload():
        """Analyze an uploaded MP4 entirely in RAM and persist only the CSV.

        Request: either the raw MP4 bytes as the request body (recommended;
        Content-Type: video/mp4) or multipart/form-data with the file in the
        "video" field ("file" is accepted as an alias). Uploads larger than
        200 MB are rejected with 413 before any processing.

        Query parameters (optional):
          - side: dominant shooting side ('auto', 'left', 'right'). Defaults
            to 'auto'. Rejects invalid values with 400.
          - feedback: whether to generate the annotated feedback video
            ('true'/'false', '1'/'0', 'yes'/'no', 'on'/'off'). Defaults to
            'true'. If false, feedback video rendering is skipped and
            video_url in the response will be null.

        Processing (all in RAM, the video never touches the filesystem):
          1. ffprobe the bytes for width/height/fps (before decoding).
          2. Stream-decode BGR frames via ffmpeg stdout, one frame at a
             time, through analisis_tiro.analizar_frames_con_cache with side.
          3. Write ONLY outputs/<id>/angulos_tiro.csv (no plot PNG, no
             video file).
          4. If feedback is enabled, re-decode the same bytes, draw the overlay
             with the existing render_feedback_frame helper, and re-encode to
             fragmented H.264 MP4 (libopenh264) into memory.

        Args: none (the video travels in the body; optional query params
            side and feedback).

        Returns: 200 JSON with id, frame_count, fps, width, height, side,
            phase_spans, rows (JSON-ready angle rows, NaN as null),
            csv (relative path), video_url (null if feedback=false) and warnings.
            The video bytes are fetched with GET video_url. Errors are JSON {"error": ...}:
            400 invalid side, not an MP4 / undecodable, 413 upload too large,
            422 zero frames decoded or no person detected ("no shooter elected").
        """
        try:
            side = request.args.get("side", "auto").strip().lower()
            if side not in ("auto", "left", "right"):
                return jsonify({"error": f"invalid side '{side}'; expected 'auto', 'left', or 'right'"}), 400
            feedback_param = request.args.get("feedback", "true").strip().lower()
            feedback = feedback_param not in ("false", "0", "no", "off")

            data, err = _read_upload_bytes()
            if err is not None:
                return err
            try:
                width, height, fps = _probe_video_bytes(data)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            try:
                df_fases, info = _analyze_upload_bytes(data, width, height, fps, side=side)
            except ValueError as exc:
                msg = str(exc)
                # WHY 422 here and 400 above: 400 means "I cannot even look
                # at this file", 422 means "I decoded it but there is no
                # analyzable content" (zero frames, no person).
                status = 422 if (
                    "zero frames decoded" in msg or "no shooter elected" in msg
                    or "no person detected" in msg
                ) else 400
                return jsonify({"error": msg}), status
            sid = _new_session_id()
            base = _uploads_base(app.config["RESULTS_DIR"])
            session_dir = os.path.join(base, sid)
            os.makedirs(session_dir, exist_ok=True)
            csv_path = os.path.join(session_dir, CSV_FILENAME)
            # WHY to_csv is the ONLY filesystem write in this handler: the
            # privacy contract allows persisting the analysis CSV and nothing
            # else. No plot PNG, no video file, no temp file anywhere.
            df_fases.to_csv(csv_path, index=False, encoding="utf-8")
            if feedback:
                try:
                    video_bytes = _render_feedback_mp4(data, width, height, info)
                except RuntimeError as exc:
                    return jsonify({"error": str(exc)}), 500
                with _VIDEO_LOCK:
                    _VIDEO_STORE[sid] = video_bytes
                video_url = "/api/analizar/%s/video" % sid
            else:
                video_url = None
            rows = _df_to_json_rows(df_fases)
            warnings = _analysis_warnings(df_fases, info)
            try:
                rel_csv = os.path.relpath(csv_path, os.getcwd())
            except ValueError:
                rel_csv = csv_path
            return jsonify({
                "id": sid,
                "frame_count": int(info["total"]),
                "fps": float(info["fps"]),
                "width": int(width),
                "height": int(height),
                "side": info.get("chosen", side),
                "phase_spans": _phase_spans(rows),
                "rows": rows,
                "csv": rel_csv,
                "video_url": video_url,
                "warnings": warnings,
            })
        except Exception as exc:  # noqa: BLE001 - WHY catch-all: the contract
            # promises JSON errors, never a traceback/HTML 500, so even an
            # unexpected bug (OOM, broken pipe) becomes a clear message.
            return jsonify({"error": "could not process video: %s" % exc}), 500

    @app.get("/api/analizar/<sid>/video")
    def analizar_video(sid):
        """Serve the in-memory feedback video of an analyzed upload.

        Args:
            sid: session id returned by POST /api/analizar.

        Returns: the fragmented H.264 MP4 bytes (probe shows
            codec_name=h264); 206 Partial Content for Range requests so the
            browser <video> element can seek; 400 for a traversal attempt,
            404 when the id is unknown (wrong id or server restarted, since
            the store is process-local memory).
        """
        error = _reject_traversal(sid)
        if error is not None:
            return error
        with _VIDEO_LOCK:
            video_bytes = _VIDEO_STORE.get(sid)
        if video_bytes is None:
            return jsonify(
                {"error": "video '%s' not found: unknown id or server "
                 "restarted (feedback videos live only in process memory; "
                 "re-upload to regenerate)" % sid}
            ), 404
        return _mp4_response(video_bytes)

    return app


def _resolve_results_dir(results_dir):
    """Absolute results directory: explicit arg, env var, or repo root.

    Args:
        results_dir: explicit directory or None to fall back to the
            ANALISIS_RESULTS_DIR environment variable and then to the repo
            root (two levels above this file).

    Returns:
        The absolute path of the directory to scan for sessions.
    """
    if results_dir:
        return os.path.abspath(results_dir)
    env = os.environ.get(RESULTS_ENV_VAR)
    if env:
        return os.path.abspath(env)
    # WHY two dirname() calls: this file lives at packages/api/app.py, so the
    # grandparent is the monorepo root regardless of the process cwd.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _reject_traversal(sid):
    """Reject session ids that could escape the results directory.

    Args:
        sid: raw `<id>` path segment from the URL.

    Returns:
        A (JSON body, 400) tuple when the id contains a separator or parent
        reference, else None meaning the id is safe to join. The check runs
        before any filesystem access, so no crafted id ever reaches open().
    """
    if not sid or "/" in sid or "\\" in sid or ".." in sid:
        return jsonify({"error": "invalid session id"}), 400
    return None


def _session_ids(results_dir):
    """Sorted ids of valid sessions: subdirs containing angulos_tiro.csv.

    Args:
        results_dir: directory to scan (one level deep, never recursive, so
            stray nested files cannot inflate the listing).

    Returns:
        Sorted list of directory names. A missing results directory yields an
        empty list, because "no runs yet" is a normal state, not an error.
    """
    if not os.path.isdir(results_dir):
        return []
    ids = []
    for name in sorted(os.listdir(results_dir)):
        if os.path.isfile(os.path.join(results_dir, name, CSV_FILENAME)):
            ids.append(name)
    return ids


def _find_video(results_dir, sid):
    """Absolute path of the session feedback video, or None when absent.

    Args:
        results_dir: the configured results directory.
        sid: already-validated session id.

    Returns:
        The first `*_feedback*.mp4` match in sorted order (a session holds at
        most one), or None so callers report absence instead of failing.
    """
    matches = sorted(glob.glob(os.path.join(results_dir, sid, VIDEO_GLOB)))
    return matches[0] if matches else None


def _read_rows(csv_path):
    """Parse the session CSV into a list of JSON-ready row dicts.

    Args:
        csv_path: absolute path of the angulos_tiro.csv file.

    Returns:
        List of dicts keyed by the exact pipeline column names, with numeric
        fields coerced to float (frame also accepts ints) and the phase left
        as a string.

    Raises:
        ValueError: when the file is empty, the header misses an expected
            column, or a data cell cannot be coerced.
    """
    # WHY utf-8-sig and not utf-8: it reads plain UTF-8 identically but also
    # strips a BOM when a spreadsheet tool added one, which utf-8 would leave
    # glued to the first column name and break header validation.
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("empty CSV, no header row")
        missing = [c for c in EXPECTED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError("missing columns: %s" % ", ".join(missing))
        rows = []
        for lineno, record in enumerate(reader, start=2):
            try:
                rows.append({
                    "frame": int(float(record["frame"])),
                    "timestamp": float(record["timestamp"]),
                    "ángulo_codo": float(record["ángulo_codo"]),
                    "ángulo_rodilla": float(record["ángulo_rodilla"]),
                    "ángulo_hombro": float(record["ángulo_hombro"]),
                    "fase_asignada": (record["fase_asignada"] or "").strip(),
                })
            except (TypeError, ValueError):
                raise ValueError("unparsable numeric cell at CSV line %d" % lineno)
    if not rows:
        raise ValueError("CSV has a header but no data rows")
    return rows


def _derive_fps(rows):
    """Frames per second from the median timestamp delta, or None.

    Args:
        rows: parsed CSV rows as returned by _read_rows.

    Returns:
        1 / median(positive timestamp diffs), or None when the timestamps
        cannot supply a period. Median (not mean) so a single timestamp
        glitch cannot skew the rate.
    """
    if len(rows) < 2:
        return None
    diffs = sorted(
        b - a for a, b in zip(
            (r["timestamp"] for r in rows), (r["timestamp"] for r in rows[1:])
        )
    )
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return None
    return 1.0 / diffs[len(diffs) // 2]


def _phase_spans(rows):
    """Per-phase frame boundaries in order of first appearance.

    Args:
        rows: parsed CSV rows as returned by _read_rows.

    Returns:
        Dict mapping each phase value to its start frame, end frame and frame
        count, so the frontend can draw the phase timeline without scanning
        every row itself.
    """
    spans = {}
    for row in rows:
        phase = row["fase_asignada"]
        frame = row["frame"]
        if phase not in spans:
            spans[phase] = {"start_frame": frame, "end_frame": frame, "frames": 1}
        else:
            spans[phase]["end_frame"] = frame
            spans[phase]["frames"] += 1
    return spans


def _summarize_session(results_dir, sid):
    """Lightweight listing entry for one session, never raising on bad data.

    Args:
        results_dir: the configured results directory.
        sid: session id (directory name).

    Returns:
        Dict with id, frame count, fps, phase spans and video/profile
        presence flags. When the CSV is corrupt the dict instead carries an
        `error` field (plus whatever presence flags could still be computed),
        so one bad run cannot fail the whole listing.
    """
    csv_path = os.path.join(results_dir, sid, CSV_FILENAME)
    video = _find_video(results_dir, sid)
    perfil_path = os.path.join(results_dir, sid, PERFIL_FILENAME)
    summary = {
        "id": sid,
        "video": os.path.basename(video) if video else None,
        "has_video": video is not None,
        "has_perfil": os.path.isfile(perfil_path),
    }
    try:
        rows = _read_rows(csv_path)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        # WHY catch-and-mark instead of letting Flask return 500: the listing
        # endpoint serves every session at once, so one corrupt CSV must not
        # take down the rest; the detail endpoint reports the same problem
        # as 422 where failing loudly is correct.
        summary["error"] = str(exc)
        summary["frame_count"] = None
        summary["fps"] = None
        summary["phase_spans"] = {}
        return summary
    summary["frame_count"] = len(rows)
    summary["fps"] = _derive_fps(rows)
    summary["phase_spans"] = _phase_spans(rows)
    return summary


def _repo_root():
    """Absolute repo root (two levels above this file: packages/api -> root).

    Args: none.

    Returns: the absolute path of the monorepo root.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _import_analysis():
    """Import the analysis core from the repo root without touching sys globally.

    WHY a lazy import instead of a top-level one: the read-only endpoints work
    with Flask alone, so a machine serving only artifacts should not pay the
    MediaPipe/matplotlib import cost (or fail) at startup; the upload path
    imports them on first use, where the error becomes a clear JSON message.

    Args: none.

    Returns: the imported analisis_tiro module.
    """
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    import analisis_tiro
    return analisis_tiro


def _uploads_base(results_dir):
    """Directory under which new upload sessions are created.

    Args:
        results_dir: the configured results directory (absolute).

    Returns: results_dir itself when it already IS an outputs directory
        (ANALISIS_RESULTS_DIR=outputs, the frontend dev setup), else
        results_dir/outputs. Either way the CSV lands at outputs/<id>/
        relative to the served tree, so the "only the CSV is persisted"
        contract holds with a single write.
    """
    if os.path.basename(os.path.normpath(results_dir)) == "outputs":
        return results_dir
    return os.path.join(results_dir, "outputs")


def _new_session_id():
    """Collision-free filesystem-safe session id for one upload.

    WHY timestamp + random hex: a timestamp alone collides on rapid double
    uploads and a bare uuid is unreadable in a directory listing; the UTC
    timestamp keeps outputs/ sortable by recency while 6 random hex digits
    make same-microsecond collisions practically impossible. Only
    alphanumerics and underscores, so it always passes _reject_traversal.

    Args: none.

    Returns: a string like tiro_20240131_120000_123456_a1b2c3.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return "tiro_%s_%s" % (ts, uuid.uuid4().hex[:6])


def _read_upload_bytes():
    """Extract the uploaded MP4 bytes from a raw body or a multipart field.

    Accepts the raw request body (recommended, Content-Type: video/mp4) or
    multipart/form-data with the file in field "video" (alias "file").
    Enforces the MAX_UPLOAD_BYTES cap on the header AND on the actual bytes.

    Args: none (reads the current Flask request).

    Returns: (bytes, None) on success, or (None, (json, status)) with a 400
        empty-body or 413 too-large JSON error tuple for the handler.
    """
    content_length = request.content_length
    if content_length is not None and content_length > MAX_UPLOAD_BYTES:
        return None, (jsonify({"error": "upload too large: got %d bytes, "
                               "limit is %d bytes (200 MB)" % (
                                   content_length, MAX_UPLOAD_BYTES)}), 413)
    data = None
    if request.files:
        for key in ("video", "file"):
            if key in request.files:
                data = request.files[key].read()
                break
    if data is None:
        # WHY get_data instead of request.data/stream: get_data caches the
        # body so a retry/debug read does not consume it, and returns b""
        # (not None) for an empty body, which the length check below turns
        # into a clear 400 instead of an obscure downstream decode error.
        data = request.get_data(cache=True) or b""
    if not data:
        return None, (jsonify({"error": "no video bytes received: POST the "
                               "MP4 as the raw request body "
                               "(Content-Type: video/mp4) or as "
                               "multipart field 'video'"}), 400)
    if len(data) > MAX_UPLOAD_BYTES:
        return None, (jsonify({"error": "upload too large: got %d bytes, "
                               "limit is %d bytes (200 MB)" % (
                                   len(data), MAX_UPLOAD_BYTES)}), 413)
    return bytes(data), None


def _parse_fps(rate_text):
    """Parse an ffprobe avg_frame_rate token into float fps, or None.

    Args:
        rate_text: string like "60/1", "30000/1001" or "0/0".

    Returns: the float rate, or None when unparseable or non-positive.
    """
    try:
        text = (rate_text or "").strip()
        if "/" in text:
            num, den = text.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(text)
    except (ValueError, ZeroDivisionError):
        return None
    if fps is None or fps != fps or fps == float("inf") or fps <= 0:
        return None
    return fps


def _probe_video_bytes(data):
    """Probe MP4 bytes with ffprobe for width, height and fps BEFORE decoding.

    WHY probe before decoding instead of reading the first decoded frame for
    the size: the rawvideo decoder needs -s WxH up front for nothing here
    (it detects them), but the ANALYSIS core and the overlay renderer need
    width/height/fps before the first frame is consumed, and probing avoids
    ever decoding a pixel just to learn the container metadata.

    Args:
        data: the uploaded MP4 bytes (never written to disk; fed to
            ffprobe stdin as pipe:0).

    Returns: (width, height, fps) with fps falling back to 30.0.

    Raises:
        ValueError: with an actionable message when the bytes are not a
            decodable MP4 (wrong file, truncated upload, or a moov-at-end
            file that pipes cannot seek: ffprobe reports width=0 with a
            "partial file" note, in which case re-exporting as faststart
            H.264 fixes it).
    """
    if not os.path.isfile(FFPROBE_BIN):
        raise ValueError("could not decode video: ffprobe not found at %s" % FFPROBE_BIN)
    try:
        proc = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
             "-of", "default=noprint_wrappers=1", "-i", "pipe:0"],
            input=data, capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("could not decode video: ffprobe timed out; "
                         "the file may not be an MP4")
    out = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    err = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    info = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip()
    try:
        width = int(info.get("width", "0"))
    except ValueError:
        width = 0
    try:
        height = int(info.get("height", "0"))
    except ValueError:
        height = 0
    fps = _parse_fps(info.get("avg_frame_rate", ""))
    if proc.returncode != 0 or width <= 0 or height <= 0:
        hint = err.strip().splitlines()
        hint_text = (" (%s)" % hint[0][:200]) if hint else ""
        if "partial file" in err:
            # WHY this specific advice: moov-at-end MP4s (like the legacy
            # cv2.VideoWriter feedback files) need seeking, which a pipe
            # cannot do, so they probe as width=0. Faststart/fragmented H.264
            # (what browsers record and what this endpoint returns) pipes
            # fine; re-exporting is the fix, not a bigger buffer.
            raise ValueError(
                "could not decode video: ffprobe could not read the stream%s; "
                "the MP4 is probably not faststart (moov atom at the end, "
                "which cannot be piped). Re-export as faststart H.264 and "
                "retry." % hint_text)
        raise ValueError("not an MP4 or undecodable video%s; send a valid "
                         ".mp4 file" % hint_text)
    if fps is None:
        # WHY 30.0 and not a failure: fps only scales timestamps; the angles
        # are per-frame, so a missing rate degrades gracefully (same fallback
        # the CLI uses for unreadable capture fps).
        fps = 30.0
    return width, height, fps


class _PipeDecoder:
    """Streaming ffmpeg MP4->rawvideo decoder yielding one BGR frame at a time.

    The input bytes are fed to ffmpeg stdin by a background thread while the
    main thread reads fixed-size BGR frames from stdout, so at most one raw
    frame (~2 MB at 924x702) plus the compressed input lives in RAM: the
    whole decoded clip (gigabytes of raw pixels) is never buffered.
    """

    def __init__(self, data, width, height):
        """Bind compressed bytes plus the probed frame geometry.

        Args:
            data: uploaded MP4 bytes fed to ffmpeg stdin (pipe:0).
            width: probed frame width in pixels.
            height: probed frame height in pixels.
        """
        self._data = data
        self._width = width
        self._height = height
        self.frames_decoded = 0
        self.stderr_text = ""
        self.returncode = None

    def __iter__(self):
        """Yield BGR frames as uint8 numpy arrays of shape (H, W, 3).

        Returns: nothing directly; yields one copied array per frame (a copy
            so downstream drawing can mutate it safely).
        """
        import numpy as np

        frame_bytes = self._width * self._height * 3
        proc = subprocess.Popen(
            [FFMPEG_BIN, "-hide_banner", "-v", "error",
             "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        captured = {}

        def _feed():
            try:
                proc.stdin.write(self._data)
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        def _drain_stderr():
            try:
                captured["err"] = proc.stderr.read()
            except (OSError, ValueError):
                captured["err"] = b""

        feeder = threading.Thread(target=_feed, daemon=True)
        drainer = threading.Thread(target=_drain_stderr, daemon=True)
        feeder.start()
        drainer.start()
        try:
            while True:
                buf = b""
                while len(buf) < frame_bytes:
                    try:
                        chunk = proc.stdout.read(frame_bytes - len(buf))
                    except (OSError, ValueError):
                        chunk = b""
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < frame_bytes:
                    break
                self.frames_decoded += 1
                # WHY .copy(): frombuffer over a throwaway bytes object is
                # read-only and dies with buf; the copy is still just one
                # frame in RAM while keeping the stream flowing.
                yield np.frombuffer(buf, dtype=np.uint8).copy().reshape(
                    (self._height, self._width, 3))
        finally:
            try:
                proc.stdout.close()
            except (OSError, ValueError):
                pass
            feeder.join(timeout=30)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            drainer.join(timeout=10)
            raw_err = captured.get("err", b"")
            self.stderr_text = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, (bytes, bytearray)) else str(raw_err)
            self.returncode = proc.returncode


def _analyze_upload_bytes(data, width, height, fps, side="auto"):
    """Decode upload bytes and run the shared analysis core over the frames.

    Args:
        data: uploaded MP4 bytes (piped to ffmpeg stdin, never to disk).
        width: probed frame width in pixels.
        height: probed frame height in pixels.
        fps: probed frames per second.
        side: dominant side ('auto', 'left', 'right') passed to analysis core.

    Returns: (df_fases, info) exactly as
        analisis_tiro.analizar_frames_con_cache returns.

    Raises:
        ValueError: "could not decode video: ..." when ffmpeg yields nothing
            usable; "zero frames decoded: ..." when the stream is empty;
            "no person detected in the clip: no shooter elected ..." when no
            frame held a valid pose (reusing the CLI election language so the
            frontend can match on it).
    """
    analisis = _import_analysis()
    decoder = _PipeDecoder(data, width, height)
    df_fases, info = analisis.analizar_frames_con_cache(
        iter(decoder), fps, width, height, side=side)
    if info.get("total", 0) == 0 or decoder.frames_decoded == 0:
        detail = decoder.stderr_text.strip().splitlines()
        hint = (" (%s)" % detail[0][:200]) if detail else ""
        raise ValueError("zero frames decoded: the file held no readable "
                         "video frames%s; send a valid .mp4" % hint)
    if decoder.returncode not in (0, None) and decoder.frames_decoded == 0:
        detail = decoder.stderr_text.strip().splitlines()
        hint = (" (%s)" % detail[0][:200]) if detail else ""
        raise ValueError("could not decode video: ffmpeg failed%s" % hint)
    if info.get("n_ok", 0) == 0:
        raise ValueError("no person detected in the clip: no shooter elected "
                         "in any frame, so there is no shot to analyze")
    return df_fases, info


def _render_feedback_mp4(data, width, height, info):
    """Re-decode the upload and encode the annotated overlay to H.264 MP4.

    WHY a second decode instead of reusing first-pass frames: the first pass
    streams (frames are consumed by inference and dropped), because buffering
    every raw frame would need gigabytes of RAM. The compressed bytes are
    already in memory, so decoding twice costs CPU but keeps memory flat, and
    mirrors generar_video_feedback's re-open of the source for its draw loop.

    Args:
        data: the same uploaded MP4 bytes (piped, never written).
        width: probed frame width in pixels.
        height: probed frame height in pixels.
        info: the info dict returned by _analyze_upload_bytes (cached pixel
            points, phases, boundaries, per-frame angles, side, fps).

    Returns: the fragmented H.264 MP4 bytes (codec_name=h264 on probe).

    Raises:
        RuntimeError: with a clear message when the encoder fails.
    """
    analisis = _import_analysis()
    fps = info.get("fps") or 30.0
    cached_points = info["cached_points"]
    phases = info["phases"]
    boundaries = info["boundaries"]
    elbow_vals = info["elbow_vals"]
    knee_vals = info["knee_vals"]
    shoulder_vals = info["shoulder_vals"]
    chosen = info["chosen"]
    total = info["total"]
    decoder = _PipeDecoder(data, width, height)

    def _render_iter():
        """Yield annotated BGR frames for the encoder stdin thread.

        Args: none (closes over the second-pass decoder and the cache).

        Returns: nothing; yields one rendered BGR frame per decoded frame,
            stopping at the shorter of the decoded stream and the cache.
        """
        for i, frame in enumerate(decoder):
            if i >= total:
                break
            pts = cached_points[i] if i < len(cached_points) else None
            angles = (
                float(elbow_vals[i]) if i < len(elbow_vals) else float("nan"),
                float(knee_vals[i]) if i < len(knee_vals) else float("nan"),
                float(shoulder_vals[i]) if i < len(shoulder_vals) else float("nan"),
            )
            phase = phases[i] if i < len(phases) else analisis.PHASE_UNKNOWN
            # WHY reuse render_feedback_frame: it draws skeleton, joint arcs,
            # HUD, phase boundaries and the timeline strip with the SAME
            # helpers as the disk path, so overlay and CSV can never disagree.
            # It copies the input, so the decoder buffer stays clean.
            yield analisis.render_feedback_frame(
                frame, pts, pts is not None, angles, phase,
                i, i / fps, chosen, phases, boundaries)

    proc = subprocess.Popen(
        [FFMPEG_BIN, "-hide_banner", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", "%dx%d" % (width, height), "-r", str(fps),
         "-i", "pipe:0", "-c:v", VIDEO_ENCODER, "-pix_fmt", "yuv420p",
         "-movflags", FRAG_MOVFLAGS, "-f", "mp4", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    captured = {}
    write_error = {}

    def _feed():
        try:
            for frame in _render_iter():
                proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            write_error["exc"] = exc
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    def _drain_stderr():
        try:
            captured["err"] = proc.stderr.read()
        except (OSError, ValueError):
            captured["err"] = b""

    feeder = threading.Thread(target=_feed, daemon=True)
    drainer = threading.Thread(target=_drain_stderr, daemon=True)
    feeder.start()
    drainer.start()
    out_chunks = bytearray()
    try:
        while True:
            try:
                chunk = proc.stdout.read(65536)
            except (OSError, ValueError):
                chunk = b""
            if not chunk:
                break
            out_chunks += chunk
    finally:
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass
        feeder.join(timeout=120)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        drainer.join(timeout=10)
    raw_err = captured.get("err", b"")
    err_text = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, (bytes, bytearray)) else str(raw_err)
    if proc.returncode != 0 or not out_chunks:
        detail = err_text.strip().splitlines()
        hint = (" (%s)" % detail[0][:300]) if detail else ""
        raise RuntimeError("could not encode feedback video%s" % hint)
    return bytes(out_chunks)


def _df_to_json_rows(df_fases):
    """Convert the phase DataFrame to JSON-ready rows (NaN becomes None).

    WHY None and not NaN: JSON has no NaN literal; Python's json would emit
    the non-standard token NaN, which strict parsers (and the frontend
    charts) reject. A null gap reads as "no detection" downstream.

    Args:
        df_fases: phase-segmented angle DataFrame from the analysis core.

    Returns: list of dicts with the exact pipeline column names.
    """
    rows = []
    for _, record in df_fases.iterrows():
        def _num(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return None if number != number else number
        rows.append({
            "frame": int(record["frame"]),
            "timestamp": float(record["timestamp"]),
            "ángulo_codo": _num(record["ángulo_codo"]),
            "ángulo_rodilla": _num(record["ángulo_rodilla"]),
            "ángulo_hombro": _num(record["ángulo_hombro"]),
            "fase_asignada": str(record["fase_asignada"]),
        })
    return rows


def _analysis_warnings(df_fases, info):
    """Collect human-readable advisory warnings for the JSON payload.

    Args:
        df_fases: phase-segmented angle DataFrame.
        info: the info dict from the analysis core (missing stats).

    Returns: list of warning strings (possibly empty); never raises.
    """
    warnings = []
    try:
        analisis = _import_analysis()
        warnings.extend(analisis.verificar_plausibilidad_fases(df_fases))
    except Exception:  # noqa: BLE001 - WHY swallow: advisories must never
        # turn a good analysis into a 500; the CSV/video already succeeded.
        pass
    try:
        missing_pct = float(info.get("missing_pct", 0.0) or 0.0)
        if missing_pct > 10.0:
            warnings.append("more than 10% of frames have no pose detection; "
                            "phase boundaries may be unreliable.")
    except (TypeError, ValueError):
        pass
    return warnings


def _mp4_response(video_bytes):
    """Build a video/mp4 response with single-range seeking support.

    WHY manual Range handling: the bytes live in process memory, not in a
    file, so Flask's send_file(conditional=True) cannot serve them; without
    206 Partial Content the browser <video> element cannot seek, which would
    defeat the feedback purpose.

    Args:
        video_bytes: the fragmented H.264 MP4 bytes.

    Returns: a Flask Response with 200 (full) or 206 (partial) status.
    """
    range_header = request.headers.get("Range", "")
    if range_header.startswith("bytes="):
        try:
            spec = range_header[len("bytes="):].strip().split(",")[0]
            start_text, _, end_text = spec.partition("-")
            total = len(video_bytes)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else total - 1
            start = max(0, min(start, total - 1))
            end = max(start, min(end, total - 1))
            chunk = video_bytes[start:end + 1]
            headers = {
                "Content-Type": "video/mp4",
                "Content-Length": str(len(chunk)),
                "Accept-Ranges": "bytes",
                "Content-Range": "bytes %d-%d/%d" % (start, end, total),
            }
            return Response(chunk, status=206, headers=headers,
                            mimetype="video/mp4")
        except (ValueError, IndexError):
            pass
    return Response(video_bytes, mimetype="video/mp4",
                    headers={"Content-Length": str(len(video_bytes)),
                             "Accept-Ranges": "bytes"})


app = create_app()


if __name__ == "__main__":
    # WHY port 5001 and not 5000: macOS reserves 5000 for AirPlay, so 5001 is
    # the Flask dev default that works on every developer machine.
    app.run(host="127.0.0.1", port=5001)
