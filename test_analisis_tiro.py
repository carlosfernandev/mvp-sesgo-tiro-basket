"""Unit tests for analisis_tiro (unittest only, no pytest dependency)."""

import contextlib
import io
import math
import sys
import os
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from analisis_tiro import (
    BASE_FPS,
    COL_ELBOW,
    COL_KNEE,
    COL_PHASE,
    COL_SHOULDER,
    COL_TIMESTAMP,
    MAX_SHOT_SPAN_MS,
    MIN_PHASE_DURATION_MS,
    PHASE_FOLLOW,
    PHASE_PREP,
    PHASE_RELEASE,
    PHASE_RISE,
    PHASE_UNKNOWN,
    arc_angles_deg,
    calcular_angulo,
    detectar_fases,
    draw_joint_arc,
    draw_phase_boundaries,
    draw_phase_strip,
    draw_skeleton,
    draw_side_chain,
    find_phase_boundaries,
    limb_angle_deg,
    phase_boundary_x,
    phase_color,
    render_feedback_frame,
    to_pixel,
    verificar_plausibilidad_fases,
)


def make_df(elbow, fps=30.0):
    """Synthetic per-frame angle DataFrame with constant knee/shoulder."""
    n = len(elbow)
    return pd.DataFrame(
        {
            "frame": list(range(n)),
            "timestamp": [i / fps for i in range(n)],
            "ángulo_codo": list(elbow),
            "ángulo_rodilla": [150.0] * n,
            "ángulo_hombro": [140.0] * n,
        }
    )


def run_with_stderr(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


class TestCalcularAngulo(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(calcular_angulo((1, 0), (0, 0), (0, 1)), 90.0)

    def test_straight_line_is_180(self):
        self.assertAlmostEqual(calcular_angulo((0, 0), (1, 0), (2, 0)), 180.0)

    def test_45_degrees(self):
        self.assertAlmostEqual(calcular_angulo((1, 0), (0, 0), (1, 1)), 45.0)

    def test_60_degrees(self):
        self.assertAlmostEqual(
            calcular_angulo((1, 0), (0, 0), (0.5, math.sqrt(3) / 2)), 60.0
        )

    def test_symmetric(self):
        a = calcular_angulo((0, 1), (0, 0), (1, 0))
        b = calcular_angulo((1, 0), (0, 0), (0, 1))
        self.assertAlmostEqual(a, b)

    def test_zero_length_first_segment_raises(self):
        with self.assertRaises(ValueError):
            calcular_angulo((1, 1), (1, 1), (2, 3))

    def test_zero_length_second_segment_raises(self):
        with self.assertRaises(ValueError):
            calcular_angulo((0, 0), (1, 1), (1, 1))

    def test_accepts_lists(self):
        self.assertAlmostEqual(
            calcular_angulo([1, 0], [0, 0], [0, 1]), 90.0
        )


class TestAspectRatio(unittest.TestCase):
    def test_pixel_conversion_recovers_true_angle(self):
        # True pixel-space geometry: angle at B is 135 degrees.
        width, height = 1920, 1080
        pix = [(0, 0), (2, 0), (3, 1)]
        expected = calcular_angulo(*pix)
        self.assertAlmostEqual(expected, 135.0, places=5)
        # Simulate MediaPipe normalized landmarks on a 16:9 frame.
        norm = [SimpleNamespace(x=x / width, y=y / height) for x, y in pix]
        recovered = [to_pixel(lm, width, height) for lm in norm]
        self.assertAlmostEqual(calcular_angulo(*recovered), 135.0, places=5)

    def test_naive_normalized_angle_is_distorted(self):
        # Proves the test above is sensitive: computing directly on
        # normalized coordinates of a non-square frame gives a wrong angle.
        width, height = 1920, 1080
        pix = [(0, 0), (2, 0), (3, 1)]
        norm_pts = [(x / width, y / height) for x, y in pix]
        naive = calcular_angulo(*norm_pts)
        self.assertGreater(abs(naive - 135.0), 5.0)

    def test_to_pixel_scales_axes_independently(self):
        lm = SimpleNamespace(x=0.5, y=0.5)
        self.assertEqual(to_pixel(lm, 1000, 500), (500.0, 250.0))


class TestDetectarFasesGoodShape(unittest.TestCase):
    # Dip at idx 2, ~90-degree crossing at idx 3, max at idx 6.
    ELBOW = [100, 80, 60, 91, 120, 160, 150, 140]

    def test_exact_boundaries(self):
        df = make_df(self.ELBOW)
        out = detectar_fases(df)
        self.assertEqual(
            list(out["fase_asignada"]),
            [
                PHASE_PREP, PHASE_PREP, PHASE_PREP,
                PHASE_RISE,
                PHASE_RELEASE, PHASE_RELEASE, PHASE_RELEASE,
                PHASE_FOLLOW,
            ],
        )

    def test_does_not_mutate_input(self):
        df = make_df(self.ELBOW)
        snapshot = df.copy(deep=True)
        out = detectar_fases(df)
        self.assertIsNot(out, df)
        pd.testing.assert_frame_equal(df, snapshot)
        self.assertNotIn("fase_asignada", df.columns)

    def test_phases_ordered_monotonically(self):
        df = make_df(self.ELBOW)
        out = detectar_fases(df)
        rank = {PHASE_PREP: 0, PHASE_RISE: 1, PHASE_RELEASE: 2, PHASE_FOLLOW: 3}
        seq = [rank[p] for p in out["fase_asignada"] if p != PHASE_UNKNOWN]
        self.assertEqual(seq, sorted(seq))


class TestDetectarFasesAdversarial(unittest.TestCase):
    def test_no_90_degree_crossing(self):
        # Interior dip (min at frame 3) but the rise stalls below 90 deg.
        df = make_df([110, 90, 65, 72, 80, 78, 76, 78])
        out, err = run_with_stderr(detectar_fases, df)
        self.assertIn("crossing", err.lower())
        phases = list(out["fase_asignada"])
        self.assertEqual(phases[0], PHASE_PREP)
        # Undefined regions must not be forced into a phase.
        self.assertTrue(all(p == PHASE_UNKNOWN for p in phases[4:]))
        self.assertNotIn(PHASE_RELEASE, phases)
        self.assertNotIn(PHASE_FOLLOW, phases)

    def test_min_after_global_max_warns_but_segments(self):
        # Global max at frame 0 precedes the dip; max is interior (frame 9)
        # so all later phases still get frames.
        df = make_df([170, 150, 130, 100, 80, 60, 70, 80, 90, 100, 95, 90])
        out, err = run_with_stderr(detectar_fases, df)
        self.assertIn("maximum", err.lower())
        self.assertIn("minimum", err.lower())
        phases = set(out["fase_asignada"])
        for expected in (PHASE_PREP, PHASE_RISE, PHASE_RELEASE, PHASE_FOLLOW):
            self.assertIn(expected, phases)

    def test_nan_gaps_kept_and_labeled(self):
        elbow = [100, 80, 60, 91, 120, 160, 150, 140]
        elbow[4] = float("nan")
        elbow[5] = float("nan")
        df = make_df(elbow)
        out, _ = run_with_stderr(detectar_fases, df)
        phases = list(out["fase_asignada"])
        self.assertEqual(phases[4], PHASE_UNKNOWN)
        self.assertEqual(phases[5], PHASE_UNKNOWN)
        self.assertEqual(phases[0], PHASE_PREP)
        self.assertEqual(phases[3], PHASE_RISE)
        # Exported values keep NaN so the user sees the gaps.
        self.assertTrue(math.isnan(out[COL_ELBOW].iloc[4]))
        self.assertTrue(math.isnan(out[COL_ELBOW].iloc[5]))

    def test_all_nan_gives_all_unknown(self):
        df = make_df([float("nan")] * 8)
        out, err = run_with_stderr(detectar_fases, df)
        self.assertIn("too few", err.lower())
        self.assertTrue((out["fase_asignada"] == PHASE_UNKNOWN).all())

    def test_single_valid_sample_gives_all_unknown(self):
        elbow = [float("nan")] * 8
        elbow[3] = 100.0
        df = make_df(elbow)
        out, err = run_with_stderr(detectar_fases, df)
        self.assertIn("too few", err.lower())
        self.assertTrue((out["fase_asignada"] == PHASE_UNKNOWN).all())

    def test_min_on_frame_zero_warns(self):
        df = make_df([60, 80, 100, 120, 150, 140, 130, 120])
        _, err = run_with_stderr(detectar_fases, df)
        self.assertIn("frame 0", err)

    def test_only_known_phase_values(self):
        df = make_df([100, 80, 60, 91, 120, 160, 150, 140])
        out = detectar_fases(df)
        allowed = {PHASE_PREP, PHASE_RISE, PHASE_RELEASE, PHASE_FOLLOW, PHASE_UNKNOWN}
        self.assertTrue(set(out["fase_asignada"]).issubset(allowed))


class TestArcAngles(unittest.TestCase):
    def test_cardinal_limb_directions(self):
        # OpenCV convention: clockwise from +x with y pointing down.
        c = (0.0, 0.0)
        self.assertAlmostEqual(limb_angle_deg(c, (1, 0)), 0.0)
        self.assertAlmostEqual(limb_angle_deg(c, (0, 1)), 90.0)
        self.assertAlmostEqual(limb_angle_deg(c, (-1, 0)), 180.0)
        self.assertAlmostEqual(limb_angle_deg(c, (0, -1)), 270.0)

    def test_right_angle_sweep(self):
        start, end = arc_angles_deg((0, 0), (1, 0), (0, 1))
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 90.0)
        self.assertAlmostEqual(end - start, 90.0)

    def test_wrap_around_adds_360(self):
        # Limbs at 350 deg and 10 deg: the clockwise sweep is 20 deg, so the
        # end must wrap to 370 instead of going the long way around.
        pa = (math.cos(math.radians(350)), math.sin(math.radians(350)))
        pb = (math.cos(math.radians(10)), math.sin(math.radians(10)))
        start, end = arc_angles_deg((0, 0), pa, pb)
        self.assertAlmostEqual(start, 350.0)
        self.assertAlmostEqual(end, 370.0)
        self.assertAlmostEqual(end - start, 20.0)

    def test_end_never_below_start(self):
        cases = [((0, 0), (1, 1), (-1, 1)), ((5, 5), (5, 0), (0, 5)),
                 ((0, 0), (0, -1), (1, 0)), ((2, 3), (-4, 3), (2, -1))]
        for joint, pa, pb in cases:
            start, end = arc_angles_deg(joint, pa, pb)
            self.assertGreaterEqual(end, start)
            self.assertLess(end - start, 360.0 + 1e-9)


class TestPhaseColors(unittest.TestCase):
    def test_all_five_phases_covered_and_distinct(self):
        colors = {
            p: phase_color(p)
            for p in (PHASE_PREP, PHASE_RISE, PHASE_RELEASE, PHASE_FOLLOW,
                      PHASE_UNKNOWN)
        }
        self.assertEqual(len(set(colors.values())), 5)
        for color in colors.values():
            self.assertEqual(len(color), 3)
            self.assertTrue(all(isinstance(v, int) and 0 <= v <= 255 for v in color))

    def test_release_is_pure_red_most_prominent(self):
        self.assertEqual(phase_color(PHASE_RELEASE), (0, 0, 255))

    def test_unknown_value_falls_back_to_gray(self):
        self.assertEqual(phase_color("no_existe"), phase_color(PHASE_UNKNOWN))


class TestBoundaryMapping(unittest.TestCase):
    def test_middle_maps_proportionally(self):
        self.assertEqual(phase_boundary_x(50, 100, 1000), int(round(50 / 99 * 999)))

    def test_first_and_last_frames_are_skipped(self):
        self.assertIsNone(phase_boundary_x(0, 100, 1000))
        self.assertIsNone(phase_boundary_x(99, 100, 1000))

    def test_monotonic_and_in_frame(self):
        xs = [phase_boundary_x(b, 100, 1000) for b in range(1, 99)]
        self.assertTrue(all(x is not None and 0 <= x < 1000 for x in xs))
        self.assertEqual(xs, sorted(xs))

    def test_find_phase_boundaries(self):
        phases = [PHASE_PREP, PHASE_PREP, PHASE_RISE, PHASE_RISE,
                  PHASE_RELEASE, PHASE_FOLLOW]
        self.assertEqual(find_phase_boundaries(phases), [2, 4, 5])
        self.assertEqual(find_phase_boundaries([PHASE_PREP] * 4), [])
        self.assertEqual(find_phase_boundaries([]), [])


def make_points_33(width=320, height=240):
    """Deterministic synthetic pixel landmarks (33 entries)."""
    rng = np.random.RandomState(7)
    xs = rng.uniform(0.3 * width, 0.7 * width, size=33)
    ys = rng.uniform(0.2 * height, 0.8 * height, size=33)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


class TestRenderHeadless(unittest.TestCase):
    """Pixel-level drawing tests on synthetic arrays (no video file needed)."""

    def test_render_with_detection_does_not_mutate_input(self):
        src = np.zeros((240, 320, 3), dtype=np.uint8)
        snapshot = src.copy()
        pts = make_points_33()
        phases = [PHASE_PREP] * 3 + [PHASE_RISE] * 3
        out = render_feedback_frame(src, pts, True, (90.0, 150.0, 140.0),
                                    PHASE_RISE, 3, 0.1, "left", phases, [3])
        np.testing.assert_array_equal(src, snapshot)
        self.assertEqual(out.shape, src.shape)
        self.assertTrue((out != 0).any())

    def test_render_without_detection_needs_no_points(self):
        src = np.zeros((240, 320, 3), dtype=np.uint8)
        phases = [PHASE_UNKNOWN] * 4
        out = render_feedback_frame(src, None, False,
                                    (float("nan"),) * 3, PHASE_UNKNOWN,
                                    0, 0.0, "left", phases, [])
        self.assertEqual(out.shape, src.shape)
        # The gray banner + ASCII banner text must leave visible pixels.
        self.assertTrue((out != 0).any())

    def test_draw_helpers_draw_in_place_and_return_frame(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        pts = make_points_33()
        for helper_out in (
            draw_skeleton(img, pts),
            draw_side_chain(img, pts, "right", (0, 0, 255)),
            draw_joint_arc(img, pts[13], pts[11], pts[15], 90.0, (0, 0, 255), 20),
            draw_phase_strip(img, [PHASE_PREP, PHASE_RISE], 0),
            draw_phase_boundaries(img, [1], 4),
        ):
            self.assertIs(helper_out, img)
        self.assertTrue((img != 0).any())

    def test_joint_arc_skips_nan_angle(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        before = img.copy()
        draw_joint_arc(img, (160.0, 120.0), (100.0, 120.0), (220.0, 120.0),
                       float("nan"), (0, 0, 255), 20)
        np.testing.assert_array_equal(img, before)


def make_phased_df(phases, fps=30.0, with_timestamp=True):
    """Phase-labeled DataFrame for verificar_plausibilidad_fases directly."""
    n = len(phases)
    data = {"frame": list(range(n)), "fase_asignada": list(phases)}
    if with_timestamp:
        data["timestamp"] = [i / fps for i in range(n)]
    return pd.DataFrame(data)


def healthy_elbow_60f():
    """60-frame dip-rise-peak-relax signal: every phase lasts > 100 ms."""
    elbow = [120 - i * (60 / 15) for i in range(16)]      # 120 -> 60 (dip)
    elbow += [60 + i * (100 / 25) for i in range(1, 26)]  # 60 -> 160 (peak)
    elbow += [160 - i * (30 / 19) for i in range(1, 20)]  # 160 -> 130
    return elbow


def one_frame_elevacion_elbow():
    """Dip followed by an immediate jump over 90 deg: 1-frame 'elevacion'."""
    return [120, 110, 100, 90, 80, 70, 60, 150, 155, 160, 165, 160,
            155, 150, 145, 140, 135, 130, 125, 120]


class TestPlausibilidadFases(unittest.TestCase):
    FPS = 30.0

    def test_healthy_shot_returns_empty_and_prints_no_warning(self):
        df = make_df(healthy_elbow_60f(), fps=self.FPS)
        out, err = run_with_stderr(detectar_fases, df)
        self.assertEqual(verificar_plausibilidad_fases(out, self.FPS), [])
        self.assertNotIn("WARNING", err)

    def test_one_frame_elevacion_is_flagged(self):
        df = make_df(one_frame_elevacion_elbow(), fps=self.FPS)
        out, err = run_with_stderr(detectar_fases, df)
        # Lock the fixture: this signal really yields a 1-frame 'elevacion'.
        self.assertEqual(list(out["fase_asignada"]).count(PHASE_RISE), 1)
        problems = verificar_plausibilidad_fases(out, self.FPS)
        self.assertTrue(problems)
        self.assertTrue(any(PHASE_RISE in p for p in problems))
        self.assertIn("WARNING", err)
        self.assertIn(PHASE_RISE, err)

    def test_total_span_longer_than_maximum_is_flagged(self):
        phases = ([PHASE_PREP] * 50 + [PHASE_RISE] * 50
                  + [PHASE_RELEASE] * 50 + [PHASE_FOLLOW] * 50)
        df = make_phased_df(phases, fps=self.FPS)
        problems = verificar_plausibilidad_fases(df, self.FPS)
        self.assertEqual(len(problems), 1)
        self.assertIn("span", problems[0].lower())

    def test_sin_deteccion_frames_are_excluded(self):
        # A single-frame 'elevacion' flags...
        df = make_phased_df([PHASE_PREP] * 9 + [PHASE_RISE], fps=self.FPS)
        self.assertTrue(verificar_plausibilidad_fases(df, self.FPS))
        # ...but the same single frame as 'sin_deteccion' does not.
        df = make_phased_df([PHASE_PREP] * 9 + [PHASE_UNKNOWN], fps=self.FPS)
        self.assertEqual(verificar_plausibilidad_fases(df, self.FPS), [])
        # 200 gap frames alone contribute no duration and no span.
        df = make_phased_df([PHASE_UNKNOWN] * 200, fps=self.FPS)
        self.assertEqual(verificar_plausibilidad_fases(df, self.FPS), [])

    def test_bad_fps_falls_back_without_raising(self):
        df = make_phased_df(
            [PHASE_PREP] * 10 + [PHASE_RISE] * 10
            + [PHASE_RELEASE] * 10 + [PHASE_FOLLOW] * 10,
            fps=self.FPS,
        )
        expected = verificar_plausibilidad_fases(df, self.FPS)
        for bad in (0, 0.0, -30.0, float("nan"), float("inf")):
            with self.subTest(fps=bad):
                self.assertEqual(verificar_plausibilidad_fases(df, bad), expected)

    def test_missing_timestamp_falls_back_to_fps(self):
        phases = ([PHASE_PREP] * 10 + [PHASE_RISE] * 10
                  + [PHASE_RELEASE] * 10 + [PHASE_FOLLOW] * 10)
        with_ts = make_phased_df(phases, fps=self.FPS, with_timestamp=True)
        without_ts = make_phased_df(phases, fps=self.FPS, with_timestamp=False)
        self.assertEqual(
            verificar_plausibilidad_fases(without_ts, self.FPS),
            verificar_plausibilidad_fases(with_ts, self.FPS),
        )

    def test_empty_dataframe_returns_empty_without_raising(self):
        self.assertEqual(verificar_plausibilidad_fases(pd.DataFrame(), 30.0), [])
        empty_cols = pd.DataFrame({"frame": [], "timestamp": [],
                                   "fase_asignada": []})
        self.assertEqual(verificar_plausibilidad_fases(empty_cols, 30.0), [])

    def test_does_not_mutate_input(self):
        df = make_phased_df(
            [PHASE_PREP] * 10 + [PHASE_RISE] * 10
            + [PHASE_RELEASE] * 10 + [PHASE_FOLLOW] * 10,
            fps=self.FPS,
        )
        snapshot = df.copy(deep=True)
        verificar_plausibilidad_fases(df, self.FPS)
        pd.testing.assert_frame_equal(df, snapshot)


if __name__ == "__main__":
    unittest.main()
