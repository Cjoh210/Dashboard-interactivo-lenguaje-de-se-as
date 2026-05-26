from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import base64
import mediapipe as mp
import asyncio
from pathlib import Path
import numpy as np
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

VIDEOS_DIR = Path("videos")
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WRIST_IDX = 0
MID_KNUCKLE_IDX = 9


@asynccontextmanager
async def lifespan(app: FastAPI):
    VIDEOS_DIR.mkdir(exist_ok=True)
    app.state.hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    logger.info("MediaPipe Hands inicializado (max_hands=%d).", MAX_HANDS)
    yield
    app.state.hands.close()
    logger.info("MediaPipe Hands cerrado.")


app = FastAPI(title="Sign Language Dashboard API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


def vectorize_hand(landmarks: list) -> list:
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)
    pts -= pts[WRIST_IDX]
    scale = float(np.linalg.norm(pts[MID_KNUCKLE_IDX]))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts[1:].flatten().tolist()


def cosine_similarity(v1: list, v2: list) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    raw = float(np.dot(a, b) / (norm_a * norm_b))
    return round((raw + 1.0) / 2.0, 4)


def _build_hand_result(hand_landmarks, handedness_label: str, reference_vector=None) -> dict:
    raw = hand_landmarks.landmark
    landmarks_2d = [{"x": float(lm.x), "y": float(lm.y)} for lm in raw]
    landmarks_3d = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in raw]
    vector_3d = vectorize_hand(landmarks_3d)
    result = {"handedness": handedness_label, "landmarks_2d": landmarks_2d, "vector_3d": vector_3d}
    if reference_vector is not None and len(reference_vector) == 60:
        result["match_score"] = cosine_similarity(vector_3d, reference_vector)
    return result


def process_image_frame(hands_model, image_bgr: np.ndarray, reference_vector=None) -> dict:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    if not results.multi_hand_landmarks:
        return {"detected": False, "hands": []}
    hands_out = []
    for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = hand_info.classification[0].label
        hands_out.append(_build_hand_result(hand_lm, label, reference_vector))
    return {"detected": True, "hands": hands_out}


def _decode_b64_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        raise ValueError("String base64 inválido.")
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return image


def _extract_video_landmarks_sync(video_path: str) -> list:
    mp_hands_local = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    frames_out = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_out.append(process_image_frame(mp_hands_local, frame))
    finally:
        cap.release()
        mp_hands_local.close()
    return frames_out


@app.post("/reference-landmarks", tags=["Landmarks"])
async def reference_landmarks(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Formato de video no soportado.")
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        frames = await asyncio.to_thread(_extract_video_landmarks_sync, tmp_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
    return {
        "total_frames": len(frames),
        "frames_with_hand": sum(1 for f in frames if f["detected"]),
        "frames": frames,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    hands_model = websocket.app.state.hands
    client = websocket.client
    logger.info("WS conectado: %s:%s", client.host, client.port)
    try:
        while True:
            payload = await websocket.receive_json()
            raw_b64 = payload.get("frame", "")
            ref_vec = payload.get("reference_vector", None)
            if not raw_b64:
                await websocket.send_json({"error": "Campo 'frame' requerido."})
                continue
            image = await asyncio.to_thread(_decode_b64_image, raw_b64)
            result = await asyncio.to_thread(process_image_frame, hands_model, image, ref_vec)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("WS desconectado: %s:%s", client.host, client.port)
    except ValueError as e:
        logger.warning("Frame inválido: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close(code=1003)
        except Exception:
            pass
    except Exception as e:
        logger.exception("Error inesperado en WS: %s", e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.get("/ping", tags=["Health"])
async def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
