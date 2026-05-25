"""
Tests unitarios y de integración.
Uso: pytest tests/test_api.py -v
"""

import base64
import json
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Ajuste de path para importar main desde la raíz del proyecto
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app, decode_image, extract_landmarks, process_frame


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def make_blank_b64(width=320, height=240) -> str:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Helpers ───────────────────────────────────────────────────────────────────
class TestDecodeImage:
    def test_with_prefix(self):
        b64 = make_blank_b64()
        img = decode_image(b64)
        assert img is not None
        assert img.shape == (240, 320, 3)

    def test_without_prefix(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        raw = base64.b64encode(buf).decode()
        img = decode_image(raw)
        assert img is not None

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError, match="base64"):
            decode_image("not_valid_base64!!!")

    def test_invalid_image_bytes_raises(self):
        garbage = base64.b64encode(b"not_an_image").decode()
        with pytest.raises(ValueError, match="decodificar"):
            decode_image(garbage)


class TestExtractLandmarks:
    def test_no_hands(self):
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        assert extract_landmarks(mock_results) == [[]]

    def test_one_hand(self):
        mock_lm = MagicMock()
        mock_lm.x, mock_lm.y, mock_lm.z = 0.1, 0.2, 0.3
        mock_hand = MagicMock()
        mock_hand.landmark = [mock_lm] * 21
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = [mock_hand]
        result = extract_landmarks(mock_results)
        assert len(result) == 1
        assert len(result[0]) == 21
        assert result[0][0] == {"x": 0.1, "y": 0.2, "z": 0.3}


# ── HTTP Endpoints ────────────────────────────────────────────────────────────
class TestPing:
    def test_ping(self, client):
        r = client.get("/ping")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestServeVideo:
    def test_file_not_found(self, client):
        r = client.get("/videos/nonexistent.mp4")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, client):
        r = client.get("/videos/../../etc/passwd")
        assert r.status_code in (403, 404)


class TestReferenceLandmarks:
    def test_missing_video_returns_404(self, client):
        r = client.get("/reference-landmarks")
        assert r.status_code == 404

    def test_with_mock_video(self, client, tmp_path, monkeypatch):
        import main as main_module

        # Crear un video de 3 frames en blanco
        video_path = tmp_path / "referencia.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 10, (320, 240))
        for _ in range(3):
            out.write(np.zeros((240, 320, 3), dtype=np.uint8))
        out.release()

        monkeypatch.setattr(main_module, "REFERENCE_VIDEO", video_path)

        r = client.get("/reference-landmarks")
        assert r.status_code == 200
        body = r.json()
        assert "total_frames" in body
        assert "landmarks" in body
        assert body["total_frames"] == 3


# ── WebSocket ─────────────────────────────────────────────────────────────────
class TestWebSocket:
    def test_blank_frame_returns_empty_landmarks(self, client):
        b64 = make_blank_b64()
        with client.websocket_connect("/ws") as ws:
            ws.send_text(b64)
            data = json.loads(ws.receive_text())
            assert data == [[]]

    def test_invalid_data_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text("INVALID_BASE64!!!")
            data = json.loads(ws.receive_text())
            assert "error" in data
