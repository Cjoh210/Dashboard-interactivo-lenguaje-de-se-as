from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import base64
import mediapipe as mp
import asyncio
from pathlib import Path
import numpy as np
import logging
from mediapipe import solutions

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
VIDEOS_DIR = Path("videos")
REFERENCE_VIDEO = VIDEOS_DIR / "referencia.mp4"
MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ── Ciclo de vida ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    VIDEOS_DIR.mkdir(exist_ok=True)
    app.state.hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    logger.info("MediaPipe Hands inicializado.")
    yield
    app.state.hands.close()
    logger.info("MediaPipe Hands cerrado.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hand Landmarks API",
    description="Detección de landmarks de manos en tiempo real con MediaPipe.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_image(data: str) -> np.ndarray:
    """Decodifica un frame base64 (con o sin prefijo data-URI) a ndarray BGR."""
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        raise ValueError("El string base64 no es válido.")
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen (formato no soportado).")
    return image


def extract_landmarks(results) -> list[list[dict]]:
    """Convierte resultados de MediaPipe a lista serializable JSON."""
    if not results.multi_hand_landmarks:
        return [[]]
    return [
        [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand.landmark]
        for hand in results.multi_hand_landmarks
    ]


def process_frame(hands_model, image: np.ndarray) -> list[list[dict]]:
    """Procesa un frame BGR y devuelve landmarks. MediaPipe requiere RGB."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    return extract_landmarks(results)


def _sync_process(hands_model, raw_data: str) -> list:
    """Función síncrona que decodifica y procesa un frame (se ejecuta en hilo)."""
    image = decode_image(raw_data)
    return process_frame(hands_model, image)


def _extract_video_landmarks(video_path: Path) -> list:
    """Extrae landmarks de todos los frames de un video (síncrono, hilo separado)."""
    mp_hands_local = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    landmarks_list = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks_list.append(process_frame(mp_hands_local, frame))
    finally:
        cap.release()
        mp_hands_local.close()

    return landmarks_list


# ── WebSocket /ws ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    hands_model = websocket.app.state.hands
    client = websocket.client
    logger.info("Cliente WebSocket conectado: %s:%s", client.host, client.port)

    try:
        while True:
            data = await websocket.receive_text()
            landmarks = await asyncio.to_thread(_sync_process, hands_model, data)
            await websocket.send_json(landmarks)

    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado: %s:%s", client.host, client.port)
    except ValueError as e:
        logger.warning("Frame inválido de %s:%s — %s", client.host, client.port, e)
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close(code=1003)
        except Exception:
            pass
    except Exception as e:
        logger.exception("Error inesperado en WebSocket: %s", e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ── GET /ping ─────────────────────────────────────────────────────────────────
@app.get("/ping", tags=["Health"], summary="Verificar estado del servidor")
async def ping():
    return {"status": "ok", "message": "pong"}


# ── GET /videos/{file_path} ───────────────────────────────────────────────────
@app.get("/videos/{file_path:path}", tags=["Static"], summary="Servir archivos de video")
async def serve_video(file_path: str):
    videos_resolved = VIDEOS_DIR.resolve()
    full_path = (VIDEOS_DIR / file_path).resolve()

    # Prevenir path traversal
    if not full_path.is_relative_to(videos_resolved):
        raise HTTPException(status_code=403, detail="Acceso denegado.")
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")

    return FileResponse(full_path)


# ── GET /reference-landmarks ──────────────────────────────────────────────────
@app.get(
    "/reference-landmarks",
    tags=["Landmarks"],
    summary="Extraer landmarks del video de referencia",
)
async def get_reference_landmarks():
    if not REFERENCE_VIDEO.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Video de referencia no encontrado en '{REFERENCE_VIDEO}'.",
        )

    try:
        landmarks_list = await asyncio.to_thread(
            _extract_video_landmarks, REFERENCE_VIDEO
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "total_frames": len(landmarks_list),
        "frames_with_hand": sum(1 for f in landmarks_list if f != [[]]),
        "landmarks": landmarks_list,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
