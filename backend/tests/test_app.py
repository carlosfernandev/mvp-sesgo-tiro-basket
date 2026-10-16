"""Unit tests for backend/src/api/app.py Flask application."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# WHY Path(__file__): tests must run from any CWD
# (`python -m unittest discover -s backend/tests -t backend`), so the
# backend/ root is anchored to this file, never to the process CWD.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd

from src.api.app import (
    _VIDEO_LOCK,
    _VIDEO_STORE,
    _analyze_upload_bytes,
    create_app,
)


class TestApp(unittest.TestCase):
    """Test suite for Flask API endpoints and contracts."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(results_dir=self.tmp_dir.name)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Clean video store between tests
        with _VIDEO_LOCK:
            _VIDEO_STORE.clear()

    def tearDown(self):
        self.tmp_dir.cleanup()
        with _VIDEO_LOCK:
            _VIDEO_STORE.clear()

    def test_health(self):
        """GET /api/health should return 200 and status ok."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("results_dir"), os.path.abspath(self.tmp_dir.name))

    def test_analizar_invalid_side_returns_400(self):
        """POST /api/analizar with invalid side returns 400 and error message."""
        # Empty body with invalid side
        res = self.client.post("/api/analizar?side=invalid")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(
            data,
            {"error": "invalid side 'invalid'; expected 'auto', 'left', or 'right'"}
        )

        # Another invalid side value
        res = self.client.post("/api/analizar?side=middle")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(
            data,
            {"error": "invalid side 'middle'; expected 'auto', 'left', or 'right'"}
        )

        # Empty string side
        res = self.client.post("/api/analizar?side=")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(
            data,
            {"error": "invalid side ''; expected 'auto', 'left', or 'right'"}
        )

    def test_analizar_side_query_params_validation(self):
        """POST /api/analizar accepts valid side values (auto, left, right)."""
        # When body is empty but side is valid, side check passes and it fails
        # at body reading with 400 'no video bytes received' instead of invalid side.
        for valid_side in ("auto", "left", "right", "LEFT", " Right ", "AUTO"):
            res = self.client.post(f"/api/analizar?side={valid_side}")
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertIn("no video bytes received", data.get("error", ""))

    @patch("src.api.app._render_feedback_mp4")
    @patch("src.api.app._analyze_upload_bytes")
    @patch("src.api.app._probe_video_bytes")
    def test_analizar_feedback_false_skips_video_rendering(
        self, mock_probe, mock_analyze, mock_render
    ):
        """POST /api/analizar?feedback=false skips video rendering and returns video_url: null."""
        mock_probe.return_value = (640, 480, 30.0)
        df_mock = pd.DataFrame([
            {
                "frame": 0,
                "timestamp": 0.0,
                "ángulo_codo": 90.0,
                "ángulo_rodilla": 120.0,
                "ángulo_hombro": 45.0,
                "fase_asignada": "preparacion",
            },
            {
                "frame": 1,
                "timestamp": 0.033,
                "ángulo_codo": 100.0,
                "ángulo_rodilla": 130.0,
                "ángulo_hombro": 55.0,
                "fase_asignada": "elevacion",
            },
        ])
        info_mock = {
            "total": 2,
            "fps": 30.0,
            "chosen": "left",
            "missing_pct": 0.0,
        }
        mock_analyze.return_value = (df_mock, info_mock)

        dummy_mp4 = b"fake-mp4-data"
        res = self.client.post(
            "/api/analizar?side=left&feedback=false",
            data=dummy_mp4,
            content_type="video/mp4",
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Verify side was forwarded
        mock_analyze.assert_called_once_with(dummy_mp4, 640, 480, 30.0, side="left")
        # Verify render was NOT called
        mock_render.assert_not_called()

        # Check response contracts
        self.assertIsNone(data.get("video_url"))
        self.assertEqual(data.get("side"), "left")
        self.assertEqual(data.get("frame_count"), 2)
        self.assertEqual(data.get("width"), 640)
        self.assertEqual(data.get("height"), 480)
        self.assertTrue(os.path.isfile(os.path.join(self.app.config["RESULTS_DIR"], data["csv"])))

        # GET video for this session returns 404
        sid = data["id"]
        res_video = self.client.get(f"/api/analizar/{sid}/video")
        self.assertEqual(res_video.status_code, 404)

    @patch("src.api.app._render_feedback_mp4")
    @patch("src.api.app._analyze_upload_bytes")
    @patch("src.api.app._probe_video_bytes")
    def test_analizar_feedback_true_renders_video(
        self, mock_probe, mock_analyze, mock_render
    ):
        """POST /api/analizar?feedback=true renders video and serves it at video_url."""
        mock_probe.return_value = (640, 480, 30.0)
        df_mock = pd.DataFrame([
            {
                "frame": 0,
                "timestamp": 0.0,
                "ángulo_codo": 90.0,
                "ángulo_rodilla": 120.0,
                "ángulo_hombro": 45.0,
                "fase_asignada": "preparacion",
            },
        ])
        info_mock = {
            "total": 1,
            "fps": 30.0,
            "chosen": "right",
            "missing_pct": 0.0,
        }
        mock_analyze.return_value = (df_mock, info_mock)
        mock_render.return_value = b"rendered-h264-mp4-stream"

        dummy_mp4 = b"fake-mp4-data"
        res = self.client.post(
            "/api/analizar?side=right&feedback=true",
            data=dummy_mp4,
            content_type="video/mp4",
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        mock_analyze.assert_called_once_with(dummy_mp4, 640, 480, 30.0, side="right")
        mock_render.assert_called_once_with(dummy_mp4, 640, 480, info_mock)

        sid = data["id"]
        expected_video_url = f"/api/analizar/{sid}/video"
        self.assertEqual(data.get("video_url"), expected_video_url)
        self.assertEqual(data.get("side"), "right")

        # GET the rendered video
        res_video = self.client.get(expected_video_url)
        self.assertEqual(res_video.status_code, 200)
        self.assertEqual(res_video.data, b"rendered-h264-mp4-stream")
        self.assertEqual(res_video.mimetype, "video/mp4")

    @patch("src.api.app._render_feedback_mp4")
    @patch("src.api.app._analyze_upload_bytes")
    @patch("src.api.app._probe_video_bytes")
    def test_analizar_feedback_falsy_values(
        self, mock_probe, mock_analyze, mock_render
    ):
        """Values ('0', 'no', 'off', 'false') all disable video feedback."""
        mock_probe.return_value = (640, 480, 30.0)
        df_mock = pd.DataFrame([
            {
                "frame": 0,
                "timestamp": 0.0,
                "ángulo_codo": 90.0,
                "ángulo_rodilla": 120.0,
                "ángulo_hombro": 45.0,
                "fase_asignada": "preparacion",
            },
        ])
        info_mock = {"total": 1, "fps": 30.0, "chosen": "right", "missing_pct": 0.0}
        mock_analyze.return_value = (df_mock, info_mock)

        for val in ("0", "no", "off", "false", "FALSE", " No "):
            mock_render.reset_mock()
            res = self.client.post(
                f"/api/analizar?feedback={val}",
                data=b"dummy",
                content_type="video/mp4",
            )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIsNone(data.get("video_url"))
            mock_render.assert_not_called()

    @patch("src.api.app._PipeDecoder")
    @patch("src.api.app._import_analysis")
    def test_analyze_upload_bytes_passes_side(self, mock_import, mock_decoder_cls):
        """_analyze_upload_bytes passes side parameter to analizar_frames_con_cache."""
        mock_analisis = MagicMock()
        mock_import.return_value = mock_analisis
        mock_decoder = MagicMock()
        mock_decoder.frames_decoded = 1
        mock_decoder.returncode = 0
        mock_decoder_cls.return_value = mock_decoder

        mock_analisis.analizar_frames_con_cache.return_value = (
            pd.DataFrame(),
            {"total": 1, "n_ok": 1, "chosen": "left"},
        )

        df, info = _analyze_upload_bytes(b"data", 640, 480, 30.0, side="left")
        mock_analisis.analizar_frames_con_cache.assert_called_once()
        _, kwargs = mock_analisis.analizar_frames_con_cache.call_args
        self.assertEqual(kwargs.get("side"), "left")

    def test_list_sesiones_empty(self):
        """GET /api/sesiones returns an empty list when no sessions exist."""
        res = self.client.get("/api/sesiones")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), [])

    def test_sesion_traversal_rejected(self):
        """Path traversal attempts in session ID must return 400."""
        res = self.client.get("/api/sesiones/..secret")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json(), {"error": "invalid session id"})

    def test_sesion_not_found(self):
        """GET /api/sesiones/<id> returns 404 for nonexistent sessions."""
        res = self.client.get("/api/sesiones/nonexistent_session")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.get_json().get("error", ""))

    def test_analizar_video_not_found(self):
        """GET /api/analizar/<id>/video returns 404 for nonexistent id."""
        res = self.client.get("/api/analizar/tiro_unknown_123/video")
        self.assertEqual(res.status_code, 404)

    def test_empty_upload_body_returns_400(self):
        """POST /api/analizar with empty body returns 400."""
        res = self.client.post("/api/analizar", data=b"")
        self.assertEqual(res.status_code, 400)
        self.assertIn("no video bytes received", res.get_json().get("error", ""))

    def test_cors_localhost(self):
        """Origin header from localhost receives CORS response headers."""
        res = self.client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
