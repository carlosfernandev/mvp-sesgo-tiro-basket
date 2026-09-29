"""Basketball shot analysis MVP: sagittal-plane joint angles + phase segmentation.

Pipeline: read a pre-recorded video -> MediaPipe PoseLandmarker (Tasks API,
VIDEO mode) -> per-frame elbow/knee/shoulder angles on the dominant side ->
smooth the elbow signal -> segment the shot into phases -> CSV + plot.

Usage:
    .venv/bin/python analisis_tiro.py shot.mp4 [--csv angulos_tiro.csv]
        [--plot perfil_angular.png] [--side {auto,left,right}]
"""

import argparse
import math
import os
import sys

import matplotlib

# A non-interactive backend: machines without a display cannot use GUI ones.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmark, PoseLandmarksConnections

# --- Module-level constants -------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join("models", "pose_landmarker_lite.task")

# Smoothing window for the elbow signal, calibrated as ~5 frames at 60 fps and
# scaled linearly with the clip fps (clamped to a minimum of 3). A centered
# rolling mean is used so extrema are not shifted in time.
BASE_SMOOTHING_WINDOW = 5
BASE_FPS = 60.0
MIN_SMOOTHING_WINDOW = 3

# Extension threshold (degrees) separating elevation from release.
EXTENSION_THRESHOLD_DEG = 90.0

# Minimum plausible duration of any single shot phase, in milliseconds. A
# phase shorter than this (for example the observed one-frame 'elevacion',
# which is ~33 ms at 30 fps) is biomechanically impossible and means the
# boundaries should not be trusted. Milliseconds are used instead of frames
# on purpose: a fixed frame count means different physical durations at
# different frame rates (5 frames are 83 ms at 60 fps but 167 ms at 30 fps),
# so a frame-based limit would be silently wrong across videos. PROVISIONAL
# guess, not measured biomechanics: replace it with a low percentile (e.g.
# p1) of phase durations from your own annotated reference shots.
MIN_PHASE_DURATION_MS = 100.0

# Maximum plausible span of a whole shot ('preparacion' start ->
# 'seguimiento' end), in milliseconds. A longer span usually means idle
# frames before/after the shot got absorbed into 'preparacion' or
# 'seguimiento', not a real slow shot. Milliseconds for the same
# fps-independence reason as above. PROVISIONAL guess, not measured
# biomechanics: replace it with a high percentile (e.g. p99) of total shot
# durations from your own annotated reference shots.
MAX_SHOT_SPAN_MS = 3000.0

# Missing-frame share above which the clip is flagged as unreliable.
MISSING_WARN_FRACTION = 0.10

# CSV / phase contract (Spanish data values, fixed by the user).
COL_FRAME = "frame"
COL_TIMESTAMP = "timestamp"
COL_ELBOW = "ángulo_codo"
COL_KNEE = "ángulo_rodilla"
COL_SHOULDER = "ángulo_hombro"
COL_PHASE = "fase_asignada"

PHASE_PREP = "preparacion"
PHASE_RISE = "elevacion"
PHASE_RELEASE = "liberacion"
PHASE_FOLLOW = "seguimiento"
PHASE_UNKNOWN = "sin_deteccion"
ALL_PHASES = (PHASE_PREP, PHASE_RISE, PHASE_RELEASE, PHASE_FOLLOW, PHASE_UNKNOWN)

# Phase colors as BGR tuples for OpenCV drawing. Each phase gets a distinct,
# readable color. 'liberacion' is pure red, the most visually prominent,
# because it is the critical phase. 'sin_deteccion' is neutral gray.
# (OpenCV uses BGR order: (blue, green, red).)
PHASE_COLORS = {
    PHASE_PREP: (255, 170, 0),  # blue
    PHASE_RISE: (0, 255, 255),  # yellow
    PHASE_RELEASE: (0, 0, 255),  # red (critical phase: most prominent)
    PHASE_FOLLOW: (0, 210, 0),  # green
    PHASE_UNKNOWN: (150, 150, 150),  # neutral gray
}

SKELETON_COLOR = (220, 220, 220)  # neutral light gray for the full skeleton
BOUNDARY_COLOR = (240, 240, 240)  # near-white vertical phase-boundary lines
TEXT_BACKING_ALPHA = 0.55
PHASE_STRIP_WIDTH = 14
# ASCII-only on purpose: cv2.putText (Hershey fonts) has no accented glyphs,
# so this label must never become "SIN DETECCIÓN".
NO_DETECTION_LABEL = "SIN DETECCION"

# Landmark triplets per side: (shoulder, elbow, wrist), (hip, knee, ankle),
# (hip, shoulder, elbow). These triplets are fixed by the user.
SIDE_JOINTS = {
    "left": {
        "shoulder": PoseLandmark.LEFT_SHOULDER,
        "elbow": PoseLandmark.LEFT_ELBOW,
        "wrist": PoseLandmark.LEFT_WRIST,
        "hip": PoseLandmark.LEFT_HIP,
        "knee": PoseLandmark.LEFT_KNEE,
        "ankle": PoseLandmark.LEFT_ANKLE,
    },
    "right": {
        "shoulder": PoseLandmark.RIGHT_SHOULDER,
        "elbow": PoseLandmark.RIGHT_ELBOW,
        "wrist": PoseLandmark.RIGHT_WRIST,
        "hip": PoseLandmark.RIGHT_HIP,
        "knee": PoseLandmark.RIGHT_KNEE,
        "ankle": PoseLandmark.RIGHT_ANKLE,
    },
}
REQUIRED_PARTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


# --- Geometry ---------------------------------------------------------------


def calcular_angulo(p1, p2, p3):
    """Angle in degrees at vertex p2 formed by segments p2->p1 and p2->p3.

    Args:
        p1, p2, p3: sequences of 2 floats (x, y) in the same coordinate space.

    Returns:
        The angle in degrees. Exactly 180.0 for collinear-forward geometry
        (p1 and p3 on opposite sides of p2, i.e. a straight line).

    Raises:
        ValueError: if either segment has zero length (p1 == p2 or p3 == p2).
    """
    v1 = (float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))
    v2 = (float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]))
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError(
            "Zero-length segment in calcular_angulo: p1=%s p2=%s p3=%s"
            % (tuple(p1), tuple(p2), tuple(p3))
        )
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    # Clamp: floating-point rounding can push the dot product just outside
    # [-1, 1], which would make acos return NaN.
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def to_pixel(landmark, width, height):
    """Convert a normalized landmark to pixel coordinates.

    MediaPipe normalizes x by frame width and y by frame height, so on a
    non-square frame one normalized unit spans a different number of pixels
    horizontally than vertically. Computing angles on normalized coordinates
    therefore distorts them (measured: ~2-4 deg error on real footage, worse
    on 16:9). All angle math must use these pixel-space points.
    """
    return (landmark.x * width, landmark.y * height)


def _smoothing_window(df_angulos):
    """Rolling-mean window scaled from BASE_SMOOTHING_WINDOW @ BASE_FPS."""
    fps = None
    if COL_TIMESTAMP in df_angulos.columns and len(df_angulos) >= 2:
        ts = pd.to_numeric(df_angulos[COL_TIMESTAMP], errors="coerce").to_numpy(
            dtype=float
        )
        diffs = np.diff(ts)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            fps = 1.0 / float(np.median(diffs))
    if fps is None or not math.isfinite(fps) or fps <= 0:
        fps = BASE_FPS
    return max(MIN_SMOOTHING_WINDOW, int(round(fps * BASE_SMOOTHING_WINDOW / BASE_FPS)))


# --- Phase segmentation -----------------------------------------------------


def verificar_plausibilidad_fases(df_fases, fps=BASE_FPS):
    """Check whether the assigned phase durations are physically plausible.

    This is an advisory coherence check, not a detector: it never changes a
    phase, it only reports durations no real shot could produce (a phase of a
    single frame, or a shot spanning most of a long clip). Every returned
    string is a human-readable problem; an empty list means everything looks
    plausible.

    Durations are frame counts converted to milliseconds using the frame
    period from the 'timestamp' column (median of the positive diffs),
    falling back to 1/fps when timestamps are absent or unusable.
    Milliseconds keep the limits fps-independent. Frames labeled
    'sin_deteccion' are excluded outright: gaps are missing data, not
    movement, so they count toward neither a phase duration nor the total
    span.

    Args:
        df_fases: DataFrame with the 'fase_asignada' column (and normally
            'timestamp'). It is only read, never mutated.
        fps: frames per second, used only when the timestamps cannot supply
            a frame period. Non-finite values and values <= 0 fall back to
            BASE_FPS instead of dividing by zero.

    Returns:
        A list of problem strings (empty when everything is plausible, when
        the DataFrame is empty, or when no analyzable phase exists). It never
        raises on bad input: unusable input yields no problems, because this
        check must never break a working segmentation.
    """
    problems = []
    if df_fases is None or len(df_fases) == 0:
        return problems
    if COL_PHASE not in df_fases.columns:
        return problems
    try:
        use_fps = float(fps)
    except (TypeError, ValueError):
        use_fps = BASE_FPS
    if not math.isfinite(use_fps) or use_fps <= 0:
        use_fps = BASE_FPS
    dt = 1.0 / use_fps
    if COL_TIMESTAMP in df_fases.columns and len(df_fases) >= 2:
        ts = pd.to_numeric(df_fases[COL_TIMESTAMP], errors="coerce").to_numpy(
            dtype=float
        )
        diffs = np.diff(ts)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            dt = float(np.median(diffs))
    phases = df_fases[COL_PHASE]
    for phase in (PHASE_PREP, PHASE_RISE, PHASE_RELEASE, PHASE_FOLLOW):
        n_frames = int((phases == phase).sum())
        if n_frames == 0:
            continue
        duration_ms = n_frames * dt * 1000.0
        if duration_ms < MIN_PHASE_DURATION_MS:
            problems.append(
                "phase '%s' lasts %.1f ms (%d frame%s), below the %.1f ms "
                "plausibility minimum; the boundaries are probably wrong, "
                "do not trust this segmentation."
                % (
                    phase,
                    duration_ms,
                    n_frames,
                    "" if n_frames == 1 else "s",
                    MIN_PHASE_DURATION_MS,
                )
            )
    analyzed = int((phases != PHASE_UNKNOWN).sum())
    if analyzed > 0:
        span_ms = analyzed * dt * 1000.0
        if span_ms > MAX_SHOT_SPAN_MS:
            problems.append(
                "total shot span is %.1f ms (%d analyzed frames), above the "
                "%.1f ms plausibility maximum; idle frames were probably "
                "absorbed into '%s'/'%s', do not trust this segmentation."
                % (span_ms, analyzed, MAX_SHOT_SPAN_MS, PHASE_PREP, PHASE_FOLLOW)
            )
    return problems


def detectar_fases(df_angulos):
    """Segment the shot into phases from the smoothed elbow-angle signal.

    Boundaries (never hardcoded frame numbers):
      preparacion: first frame -> frame of the elbow-angle MINIMUM (dip).
      elevacion:   that minimum -> first frame AFTER it crossing ~90 deg.
      liberacion:  the ~90-degree crossing -> frame of the elbow MAXIMUM.
      seguimiento: that maximum -> last frame.

    Frames whose elbow angle is NaN (no pose detected) are labeled
    'sin_deteccion' instead of being forced into a phase, and NaN is only
    filled on an internal search copy so the exported values keep the gaps.

    Args:
        df_angulos: DataFrame with at least the 'ángulo_codo' column (and
            normally 'frame' + 'timestamp').

    Returns:
        A NEW DataFrame (input is not mutated) with the 'fase_asignada' column.
        Guards: every failed assumption is reported as a WARNING on stderr.
    """
    df = df_angulos.copy()
    n = len(df)
    phases = pd.Series([PHASE_UNKNOWN] * n)

    # NOTE: the duration-plausibility check (verificar_plausibilidad_fases)
    # runs only on the success path at the end of this function. Each early
    # return below already prints a hard-failure WARNING and yields
    # 'sin_deteccion' (or a 'preparacion'-only stub), where phase durations
    # are meaningless, so there is nothing plausible to check there. Skipping
    # those paths is deliberate, not an oversight.
    if n == 0:
        print("WARNING: detectar_fases received an empty DataFrame.", file=sys.stderr)
        df[COL_PHASE] = phases
        return df

    elbow = pd.to_numeric(df[COL_ELBOW], errors="coerce")
    valid_mask = elbow.notna()
    n_valid = int(valid_mask.sum())

    if n_valid < 3:
        print(
            "WARNING: too few valid elbow samples (%d of %d frames); "
            "no phase boundaries can be estimated. All frames -> '%s'."
            % (n_valid, n, PHASE_UNKNOWN),
            file=sys.stderr,
        )
        df[COL_PHASE] = phases
        return df

    window = _smoothing_window(df)
    # Internal search copy only: fill NaN so extrema are well defined. The
    # exported column keeps NaN so the user can see the detection gaps.
    filled = elbow.ffill().bfill()
    smooth = filled.rolling(window, center=True, min_periods=1).mean()

    i_min = int(smooth.idxmin())
    if not math.isfinite(smooth.iloc[i_min]):
        print(
            "WARNING: no usable elbow-angle minimum found; "
            "all frames -> '%s'." % PHASE_UNKNOWN,
            file=sys.stderr,
        )
        df[COL_PHASE] = phases
        return df

    if i_min == 0:
        print(
            "WARNING: elbow minimum lands on frame 0, so phase "
            "'%s' is degenerate (single frame)." % PHASE_PREP,
            file=sys.stderr,
        )

    # Inconsistent-signal guard: a global maximum BEFORE the minimum means the
    # clip does not show the expected dip-then-extend shape.
    i_global_max = int(smooth.idxmax())
    if i_global_max < i_min:
        print(
            "WARNING: global elbow maximum (frame %d) precedes the minimum "
            "(frame %d): inconsistent signal, the clip may not contain a "
            "single dip-then-extend shot." % (i_global_max, i_min),
            file=sys.stderr,
        )

    # The release maximum must be searched STRICTLY AFTER the minimum.
    tail = smooth.iloc[i_min + 1 :]
    tail_valid = tail[tail.notna()]
    if tail_valid.empty:
        print(
            "WARNING: no frames after the elbow minimum (frame %d); "
            "cannot estimate release/follow-through." % i_min,
            file=sys.stderr,
        )
        ok = valid_mask.to_numpy()
        phases.iloc[: i_min + 1] = PHASE_PREP
        phases = phases.where(ok, PHASE_UNKNOWN)
        df[COL_PHASE] = phases.values
        return df
    i_max = int(tail_valid.idxmax())

    # First frame AFTER the minimum reaching/crossing ~90 deg on the rise.
    rising = smooth.iloc[i_min + 1 :]
    crossing = rising[rising >= EXTENSION_THRESHOLD_DEG]
    if crossing.empty:
        print(
            "WARNING: no %.0f-degree crossing on the rising edge after the "
            "minimum (frame %d); elevation/release/follow-through are "
            "undefined, later frames -> '%s'."
            % (EXTENSION_THRESHOLD_DEG, i_min, PHASE_UNKNOWN),
            file=sys.stderr,
        )
        ok = valid_mask.to_numpy()
        phases.iloc[: i_min + 1] = PHASE_PREP
        phases = phases.where(ok, PHASE_UNKNOWN)
        df[COL_PHASE] = phases.values
        return df
    i_cross = int(crossing.index[0])

    if not (i_min < i_cross <= i_max):
        print(
            "WARNING: inconsistent boundaries (min=%d, 90-deg crossing=%d, "
            "max=%d); later frames -> '%s'."
            % (i_min, i_cross, i_max, PHASE_UNKNOWN),
            file=sys.stderr,
        )
        ok = valid_mask.to_numpy()
        phases.iloc[: i_min + 1] = PHASE_PREP
        phases = phases.where(ok, PHASE_UNKNOWN)
        df[COL_PHASE] = phases.values
        return df

    ok = valid_mask.to_numpy()
    idx = np.arange(n)
    phases.iloc[(idx >= 0) & (idx <= i_min)] = PHASE_PREP
    phases.iloc[(idx > i_min) & (idx <= i_cross)] = PHASE_RISE
    phases.iloc[(idx > i_cross) & (idx <= i_max)] = PHASE_RELEASE
    phases.iloc[idx > i_max] = PHASE_FOLLOW
    # Frames without detection keep no forced phase.
    phases = phases.where(ok, PHASE_UNKNOWN)

    print(
        "Phase boundaries (frames): %s ends at %d, %s ends at %d, "
        "%s ends at %d."
        % (PHASE_PREP, i_min, PHASE_RISE, i_cross, PHASE_RELEASE, i_max)
    )
    # Duration-plausibility advisory: warning only, never a correction. This
    # loop prints to stderr and cannot re-assign, suppress, or otherwise alter
    # the phases decided above; its job is to say "do not trust this output".
    df[COL_PHASE] = phases.values
    for problem in verificar_plausibilidad_fases(df):
        print("WARNING: " + problem, file=sys.stderr)
    return df


# --- Video processing -------------------------------------------------------


def _select_side(frames_landmarks, side):
    """Choose the dominant side from mean landmark visibility.

    In a lateral (sagittal) view MediaPipe's LEFT/RIGHT labels are anatomical,
    not image-relative, so the side facing the camera is the one whose 6
    required landmarks are seen more confidently on average.
    """
    if side in ("left", "right"):
        print("Using side '%s' (CLI override)." % side)
        return side
    scores = {}
    counts = {}
    for entry in frames_landmarks:
        if entry is None:
            continue
        for candidate in ("left", "right"):
            parts = SIDE_JOINTS[candidate]
            for part in REQUIRED_PARTS:
                lm = entry[parts[part].value]
                scores[candidate] = scores.get(candidate, 0.0) + float(lm.visibility)
                counts[candidate] = counts.get(candidate, 0) + 1
    means = {
        candidate: (scores.get(candidate, 0.0) / counts[candidate] if counts.get(candidate) else 0.0)
        for candidate in ("left", "right")
    }
    chosen = "left" if means["left"] >= means["right"] else "right"
    print(
        "Auto-selected side '%s' (mean visibility left=%.3f right=%.3f "
        "over frames with detection)." % (chosen, means["left"], means["right"])
    )
    return chosen


def procesar_video(ruta_video, ruta_csv="angulos_tiro.csv", ruta_plot="perfil_angular.png",
                   side="auto", model_path=DEFAULT_MODEL_PATH):
    """Run the full pipeline on a pre-recorded video.

    Reads frames, runs pose inference, extracts sagittal-plane angles on the
    dominant side, segments phases, writes the CSV and the plot, and prints a
    console summary. Exits non-zero if the video cannot be opened.
    """
    import cv2

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("ERROR: cannot open video '%s'." % ruta_video, file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not math.isfinite(fps) or fps <= 0:
        print(
            "WARNING: unreadable fps (got %r); falling back to 30 fps for "
            "timestamps." % (fps,),
            file=sys.stderr,
        )
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    frames_landmarks = []
    frame_index = 0
    prev_timestamp_ms = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Timestamps must be strictly monotonically increasing integers.
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            if prev_timestamp_ms is not None:
                timestamp_ms = max(timestamp_ms, prev_timestamp_ms + 1)
            prev_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                frames_landmarks.append(list(result.pose_landmarks[0]))
            else:
                frames_landmarks.append(None)
            frame_index += 1
    finally:
        cap.release()
        landmarker.close()

    total = len(frames_landmarks)
    n_ok = sum(1 for entry in frames_landmarks if entry is not None)
    n_missing = total - n_ok
    missing_pct = (100.0 * n_missing / total) if total else 0.0

    chosen = _select_side(frames_landmarks, side)
    parts = SIDE_JOINTS[chosen]

    rows = []
    for i, entry in enumerate(frames_landmarks):
        if entry is None:
            elbow = knee = shoulder = float("nan")
        else:
            s_px = to_pixel(entry[parts["shoulder"].value], width, height)
            e_px = to_pixel(entry[parts["elbow"].value], width, height)
            w_px = to_pixel(entry[parts["wrist"].value], width, height)
            h_px = to_pixel(entry[parts["hip"].value], width, height)
            k_px = to_pixel(entry[parts["knee"].value], width, height)
            a_px = to_pixel(entry[parts["ankle"].value], width, height)
            elbow = calcular_angulo(s_px, e_px, w_px)
            knee = calcular_angulo(h_px, k_px, a_px)
            shoulder = calcular_angulo(h_px, s_px, e_px)
        rows.append(
            {
                COL_FRAME: i,
                COL_TIMESTAMP: i / fps,
                COL_ELBOW: elbow,
                COL_KNEE: knee,
                COL_SHOULDER: shoulder,
            }
        )

    df = pd.DataFrame(rows, columns=[COL_FRAME, COL_TIMESTAMP, COL_ELBOW, COL_KNEE, COL_SHOULDER])
    df_fases = detectar_fases(df)
    # pandas and matplotlib both raise if the parent directory is missing.
    for ruta in (ruta_csv, ruta_plot):
        parent = os.path.dirname(os.path.abspath(ruta))
        if parent:
            os.makedirs(parent, exist_ok=True)
    df_fases.to_csv(ruta_csv, index=False, encoding="utf-8")

    _plot_perfil(df_fases, ruta_plot)

    print("Total frames: %d" % total)
    print("Frames with detection: %d" % n_ok)
    print("Missing frames: %d (%.1f%%)" % (n_missing, missing_pct))
    if missing_pct > 100.0 * MISSING_WARN_FRACTION:
        print(
            "WARNING: more than %.0f%% of frames have no pose detection; "
            "phase boundaries may be unreliable." % (100.0 * MISSING_WARN_FRACTION),
            file=sys.stderr,
        )
    for col in (COL_ELBOW, COL_KNEE, COL_SHOULDER):
        series = pd.to_numeric(df_fases[col], errors="coerce")
        if series.notna().any():
            print("%s: min=%.1f max=%.1f deg" % (col, series.min(), series.max()))
        else:
            print("%s: n/a (no valid samples)" % col)
    return df_fases


def _plot_perfil(df_fases, ruta_plot):
    """Plot the three angle signals vs time with phase-boundary lines."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    t = pd.to_numeric(df_fases[COL_TIMESTAMP], errors="coerce").to_numpy(dtype=float)
    for col in (COL_ELBOW, COL_KNEE, COL_SHOULDER):
        y = pd.to_numeric(df_fases[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(t, y, label=col)
    if COL_PHASE in df_fases.columns and len(df_fases) > 1:
        phase = df_fases[COL_PHASE].to_numpy()
        changes = np.where(phase[1:] != phase[:-1])[0]
        for j in changes:
            # Boundary between frame j and j+1; draw at the later timestamp and
            # name the phase that starts there, so the plot can be read directly.
            ax.axvline(t[j + 1], linestyle="--", alpha=0.7)
            ax.annotate(
                str(phase[j + 1]),
                xy=(t[j + 1], ax.get_ylim()[1]),
                xytext=(2, -10),
                textcoords="offset points",
                rotation=90,
                fontsize=7,
                va="top",
                ha="left",
            )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (deg)")
    ax.set_title("Shot angular profile")
    ax.legend()
    fig.tight_layout()
    ax.margins(x=0.02)
    fig.savefig(ruta_plot)
    plt.close(fig)
    print("Plot saved to '%s'." % ruta_plot)


# --- Annotated-video feedback overlay -----------------------------------------
# These helpers draw the visual feedback burned into the output MP4. They use
# the SAME angle math (calcular_angulo on pixel points) and the SAME
# detectar_fases() call as the CSV path, so the overlay and the CSV can never
# disagree. All drawing helpers take a BGR frame, draw in place, and return
# it; render_feedback_frame() copies first so the caller's array is never
# mutated. All text passed to cv2.putText MUST be pure ASCII (Hershey fonts
# have no glyphs for accented characters).


def phase_color(phase):
    """BGR color tuple for a phase value; unknown values fall back to gray."""
    return PHASE_COLORS.get(phase, PHASE_COLORS[PHASE_UNKNOWN])


def limb_angle_deg(center, point):
    """Direction of the limb center->point in degrees, normalized to [0, 360).

    Uses math.atan2(dy, dx), which matches the OpenCV convention (angles
    measured clockwise from +x with y pointing down), so the result can be
    fed directly to cv2.ellipse.
    """
    angle = math.degrees(
        math.atan2(float(point[1]) - float(center[1]), float(point[0]) - float(center[0]))
    )
    if angle < 0.0:
        angle += 360.0
    return angle


def arc_angles_deg(joint, limb_a, limb_b):
    """Start/end degrees for cv2.ellipse spanning the joint angle.

    Returns (start, end) with end >= start: when the raw end angle wraps
    below start, 360 is added so the ellipse draws the clockwise sweep from
    limb A to limb B instead of going the long way around.
    """
    start = limb_angle_deg(joint, limb_a)
    end = limb_angle_deg(joint, limb_b)
    if end < start:
        end += 360.0
    return (start, end)


def find_phase_boundaries(phases):
    """Frame indices where the phase label changes (boundary j starts at j)."""
    return [j for j in range(1, len(phases)) if phases[j] != phases[j - 1]]


def phase_boundary_x(boundary_frame, total_frames, frame_width):
    """X pixel of a phase-boundary vertical line, or None to skip it.

    Boundaries are temporal, so they are mapped proportionally across the
    frame width: x = boundary_frame / (total_frames - 1) * (frame_width - 1).
    A boundary on the first or last frame would land exactly on (or off) the
    frame edge, so it is skipped (None) instead of drawing a line off-frame.
    """
    if total_frames <= 1 or frame_width <= 0:
        return None
    if boundary_frame <= 0 or boundary_frame >= total_frames - 1:
        return None
    return int(round(boundary_frame / (total_frames - 1) * (frame_width - 1)))


def _draw_text_with_backing(img, text, org, color, font_scale=1.0, thickness=2):
    """Draw ASCII text with a dark translucent backing rectangle for contrast."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    pad = 6
    x0 = max(0, x - pad)
    y0 = max(0, y - th - pad)
    x1 = min(img.shape[1], x + tw + pad)
    y1 = min(img.shape[0], y + baseline + pad)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, TEXT_BACKING_ALPHA, roi, 1.0 - TEXT_BACKING_ALPHA, 0, roi)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def draw_joint_arc(img, joint_px, limb_a_px, limb_b_px, angle_deg, color, radius):
    """Draw the joint-angle arc (cv2.ellipse) plus a numeric degree label.

    The label is placed along the arc bisector, just outside the arc radius,
    so it does not overlap the limbs. NaN angles draw nothing (no fake
    numbers). The label uses the ASCII suffix "deg" (no degree sign: Hershey
    fonts cannot render it).
    """
    import cv2

    if angle_deg is None or (
        isinstance(angle_deg, float) and math.isnan(angle_deg)
    ):
        return img
    start_deg, end_deg = arc_angles_deg(joint_px, limb_a_px, limb_b_px)
    cx, cy = int(round(joint_px[0])), int(round(joint_px[1]))
    radius = max(1, int(radius))
    cv2.ellipse(
        img, (cx, cy), (radius, radius), 0.0, start_deg, end_deg, color, 2, cv2.LINE_AA
    )
    mid_rad = math.radians((start_deg + end_deg) / 2.0)
    gap = radius + 16
    lx = int(round(cx + gap * math.cos(mid_rad)))
    ly = int(round(cy + gap * math.sin(mid_rad)))
    label = "%.1f deg" % float(angle_deg)
    font_scale, thickness = 0.55, 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    lx = max(0, min(lx, img.shape[1] - tw - 1))
    ly = max(th + 1, min(ly, img.shape[0] - 1))
    _draw_text_with_backing(img, label, (lx, ly), (255, 255, 255), font_scale, thickness)
    return img


def draw_skeleton(img, points_px, color=SKELETON_COLOR, thickness=1):
    """Draw all 35 POSE_LANDMARKS connections thin in a neutral light color."""
    import cv2

    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        a, b = int(conn.start), int(conn.end)
        if 0 <= a < len(points_px) and 0 <= b < len(points_px):
            pa = (int(round(points_px[a][0])), int(round(points_px[a][1])))
            pb = (int(round(points_px[b][0])), int(round(points_px[b][1])))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    return img


# Thick dominant-side chain: shoulder->elbow->wrist, hip->knee->ankle, hip->shoulder.
DOMINANT_CHAIN = (
    ("shoulder", "elbow"),
    ("elbow", "wrist"),
    ("hip", "knee"),
    ("knee", "ankle"),
    ("hip", "shoulder"),
)


def draw_side_chain(img, points_px, side, color, thickness=3):
    """Draw the analyzed (dominant) side chain thick in the phase color.

    Marks the 6 joints with small filled circles. The side MUST be the same
    one the angles were computed from (see _select_side).
    """
    import cv2

    parts = SIDE_JOINTS[side]
    idx = {part: parts[part].value for part in REQUIRED_PARTS}
    for a_part, b_part in DOMINANT_CHAIN:
        pa = (int(round(points_px[idx[a_part]][0])), int(round(points_px[idx[a_part]][1])))
        pb = (int(round(points_px[idx[b_part]][0])), int(round(points_px[idx[b_part]][1])))
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    for part in REQUIRED_PARTS:
        p = (int(round(points_px[idx[part]][0])), int(round(points_px[idx[part]][1])))
        cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)
    return img


def draw_phase_boundaries(img, boundaries, total_frames, color=BOUNDARY_COLOR):
    """Draw a full-height vertical line per phase boundary (neutral color).

    Boundary frame j maps to an x pixel via phase_boundary_x(); boundaries on
    the first/last frame are skipped (they would fall off-frame).
    """
    import cv2

    h, w = img.shape[0], img.shape[1]
    for b in boundaries:
        x = phase_boundary_x(b, total_frames, w)
        if x is None:
            continue
        cv2.line(img, (x, 0), (x, h - 1), color, 2, cv2.LINE_AA)
    return img


def draw_phase_strip(img, phases, current_idx, strip_width=PHASE_STRIP_WIDTH):
    """Thin vertical color strip down the LEFT EDGE: a mini phase timeline.

    Every frame of the clip gets a proportional vertical segment in its phase
    color, so the whole shot timeline reads at a glance; the current frame's
    position is marked with a white dot just right of the strip.
    """
    import cv2

    h = img.shape[0]
    n = len(phases)
    if n == 0:
        return img
    for i, ph in enumerate(phases):
        y0 = int(i * h / n)
        y1 = max(y0 + 1, int((i + 1) * h / n))
        cv2.rectangle(img, (0, y0), (strip_width, y1), phase_color(ph), -1)
    y = int(current_idx / (n - 1) * (h - 1)) if n > 1 else 0
    y = max(0, min(y, h - 1))
    cv2.circle(img, (strip_width + 8, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def draw_hud(img, frame_idx, timestamp_s, side, phase):
    """Large phase name top-left in the phase color plus a compact HUD line."""
    color = phase_color(phase)
    _draw_text_with_backing(img, str(phase).upper(), (24, 42), color, 1.0, 2)
    hud = "frame %d | t=%.2f s | side %s" % (int(frame_idx), float(timestamp_s), side)
    _draw_text_with_backing(img, hud, (24, 78), (255, 255, 255), 0.6, 1)
    return img


def draw_no_detection(img, frame_idx, timestamp_s):
    """Clearly visible missing-pose state: gray band + ASCII banner + HUD."""
    import cv2

    h, w = img.shape[0], img.shape[1]
    band_h = min(h, 110)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, band_h), PHASE_COLORS[PHASE_UNKNOWN], -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    _draw_text_with_backing(
        img, NO_DETECTION_LABEL, (24, 42), (255, 255, 255), 1.0, 2
    )
    hud = "frame %d | t=%.2f s | no pose" % (int(frame_idx), float(timestamp_s))
    _draw_text_with_backing(img, hud, (24, 78), (255, 255, 255), 0.6, 1)
    return img


def render_feedback_frame(frame_bgr, points_px, detected, angles_deg, phase,
                          frame_idx, timestamp_s, side, phases_all, boundaries):
    """Render one annotated frame on a COPY of frame_bgr (input not mutated).

    Args:
        frame_bgr: source BGR frame (OpenCV order, as VideoWriter wants it).
        points_px: cached pixel coordinates for all landmarks, or None.
        detected: per-frame pose-detected flag from the inference pass.
        angles_deg: (elbow, knee, shoulder) in degrees (may hold NaN).
        phase: phase label for this frame (from detectar_fases).
        frame_idx / timestamp_s: position of this frame in the clip.
        side: dominant side the angles were computed from.
        phases_all: full-clip phase list (for the timeline strip).
        boundaries: frame indices where the phase changes.
    """
    img = frame_bgr.copy()
    h, w = img.shape[0], img.shape[1]
    if detected and points_px is not None:
        color = phase_color(phase)
        draw_skeleton(img, points_px)
        draw_side_chain(img, points_px, side, color)
        parts = SIDE_JOINTS[side]
        idx = {part: parts[part].value for part in REQUIRED_PARTS}
        radius = max(12, int(min(w, h) * 0.035))
        elbow, knee, shoulder = angles_deg
        draw_joint_arc(img, points_px[idx["elbow"]],
                       points_px[idx["shoulder"]], points_px[idx["wrist"]],
                       elbow, color, radius)
        draw_joint_arc(img, points_px[idx["knee"]],
                       points_px[idx["hip"]], points_px[idx["ankle"]],
                       knee, color, radius)
        draw_joint_arc(img, points_px[idx["shoulder"]],
                       points_px[idx["hip"]], points_px[idx["elbow"]],
                       shoulder, color, radius)
        draw_hud(img, frame_idx, timestamp_s, side, phase)
    else:
        # No skeleton, no fake numbers: detection gaps must read as gaps.
        draw_no_detection(img, frame_idx, timestamp_s)
    draw_phase_boundaries(img, boundaries, len(phases_all))
    draw_phase_strip(img, phases_all, frame_idx)
    return img


def generar_video_feedback(ruta_video, ruta_salida, side="auto",
                           model_path=DEFAULT_MODEL_PATH, codec="mp4v"):
    """Render a playable copy of the input video with visual feedback burned in.

    Two loops, ONE inference pass (single source of truth for phases):
      Loop 1: run MediaPipe inference frame by frame; cache per-frame
        landmark pixel coordinates (to_pixel for all landmarks) plus a
        per-frame pose-detected flag. No second inference ever runs.
      Then: pick the dominant side with _select_side(), compute the three
        angles per frame with calcular_angulo (the same math procesar_video
        uses), build the same DataFrame, and call detectar_fases() — the
        exact same call the CSV path makes — so overlay and CSV agree.
      Loop 2: re-open the source video and draw each frame from the CACHED
        coordinates + phase labels, writing the annotated MP4.

    Drawing happens on the original BGR frames (never on the RGB conversion
    MediaPipe needs); the writer preserves the source frame size and fps.

    Returns:
        The phase DataFrame (same contract as procesar_video), so callers can
        assert CSV/overlay consistency directly.
    """
    import cv2

    # --- Loop 1: single inference pass + coordinate cache -------------------
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("ERROR: cannot open video '%s'." % ruta_video, file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not math.isfinite(fps) or fps <= 0:
        print(
            "WARNING: unreadable fps (got %r); falling back to 30 fps for "
            "timestamps." % (fps,),
            file=sys.stderr,
        )
        fps = 30.0
    # Fractional source fps (e.g. 29.97) is preserved as-is: VideoWriter
    # accepts it, and rounding it would shift every timestamp.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    raw_landmarks = []  # landmark objects per frame (None when undetected)
    cached_points = []  # pixel coords per frame (None when undetected)
    frame_index = 0
    prev_timestamp_ms = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Timestamps must be strictly monotonically increasing integers.
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            if prev_timestamp_ms is not None:
                timestamp_ms = max(timestamp_ms, prev_timestamp_ms + 1)
            prev_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                entry = list(result.pose_landmarks[0])
                raw_landmarks.append(entry)
                cached_points.append([to_pixel(lm, width, height) for lm in entry])
            else:
                raw_landmarks.append(None)
                cached_points.append(None)
            frame_index += 1
    finally:
        cap.release()
        landmarker.close()

    total = len(raw_landmarks)
    if total == 0:
        print("WARNING: no frames read from '%s'." % ruta_video, file=sys.stderr)

    # --- Same angles + same phase call as the CSV path ----------------------
    chosen = _select_side(raw_landmarks, side)
    parts = SIDE_JOINTS[chosen]

    rows = []
    for i, pts in enumerate(cached_points):
        if pts is None:
            elbow = knee = shoulder = float("nan")
        else:
            s_px = pts[parts["shoulder"].value]
            e_px = pts[parts["elbow"].value]
            w_px = pts[parts["wrist"].value]
            h_px = pts[parts["hip"].value]
            k_px = pts[parts["knee"].value]
            a_px = pts[parts["ankle"].value]
            elbow = calcular_angulo(s_px, e_px, w_px)
            knee = calcular_angulo(h_px, k_px, a_px)
            shoulder = calcular_angulo(h_px, s_px, e_px)
        rows.append(
            {
                COL_FRAME: i,
                COL_TIMESTAMP: i / fps,
                COL_ELBOW: elbow,
                COL_KNEE: knee,
                COL_SHOULDER: shoulder,
            }
        )

    df = pd.DataFrame(rows, columns=[COL_FRAME, COL_TIMESTAMP, COL_ELBOW, COL_KNEE, COL_SHOULDER])
    df_fases = detectar_fases(df)
    phases = list(df_fases[COL_PHASE])
    boundaries = find_phase_boundaries(phases)
    elbow_vals = pd.to_numeric(df_fases[COL_ELBOW], errors="coerce").to_numpy(dtype=float)
    knee_vals = pd.to_numeric(df_fases[COL_KNEE], errors="coerce").to_numpy(dtype=float)
    shoulder_vals = pd.to_numeric(df_fases[COL_SHOULDER], errors="coerce").to_numpy(dtype=float)

    # --- Loop 2: draw from cache, never re-running inference ----------------
    cap2 = cv2.VideoCapture(ruta_video)
    if not cap2.isOpened():
        print("ERROR: cannot re-open video '%s'." % ruta_video, file=sys.stderr)
        sys.exit(1)
    parent = os.path.dirname(os.path.abspath(ruta_salida))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))
    if not out.isOpened():
        cap2.release()
        raise RuntimeError(
            "Cannot open VideoWriter with codec '%s' for '%s'. "
            "This machine may lack that encoder; retry with --codec mp4v."
            % (codec, ruta_salida)
        )
    written = 0
    try:
        for i in range(total):
            ok, frame = cap2.read()
            if not ok or frame is None:
                print(
                    "WARNING: source ended early at frame %d of %d; "
                    "wrote %d frames." % (i, total, written),
                    file=sys.stderr,
                )
                break
            pts = cached_points[i]
            img = render_feedback_frame(
                frame, pts, pts is not None,
                (elbow_vals[i], knee_vals[i], shoulder_vals[i]),
                phases[i], i, i / fps, chosen, phases, boundaries,
            )
            out.write(img)
            written += 1
    finally:
        cap2.release()
        out.release()

    print(
        "Feedback video saved to '%s' (%d/%d frames, %dx%d @ %.2f fps, "
        "codec %s)." % (ruta_salida, written, total, width, height, fps, codec)
    )
    return df_fases


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract sagittal-plane joint angles from a basketball "
        "shot video and segment the shot into phases."
    )
    parser.add_argument("video", help="Path to the pre-recorded video file.")
    parser.add_argument("--csv", default="angulos_tiro.csv", help="Output CSV path.")
    parser.add_argument("--plot", default="perfil_angular.png", help="Output plot path.")
    parser.add_argument("--side", choices=("auto", "left", "right"), default="auto",
                        help="Dominant side; 'auto' picks the side with higher "
                        "mean landmark visibility.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to the pose_landmarker .task model file.")
    parser.add_argument("--video-feedback", default=None, metavar="RUTA",
                        help="Optional output path for an annotated MP4 with "
                        "skeleton, joint angles and phases burned in. "
                        "CSV/plot behavior is unchanged when absent.")
    parser.add_argument("--codec", default="mp4v",
                        help="FourCC codec for --video-feedback (default mp4v, "
                        "which works here; try avc1 on machines with an H.264 "
                        "encoder).")
    args = parser.parse_args(argv)
    procesar_video(args.video, ruta_csv=args.csv, ruta_plot=args.plot,
                   side=args.side, model_path=args.model)
    if args.video_feedback:
        generar_video_feedback(args.video, args.video_feedback,
                               side=args.side, model_path=args.model,
                               codec=args.codec)


if __name__ == "__main__":
    main()
