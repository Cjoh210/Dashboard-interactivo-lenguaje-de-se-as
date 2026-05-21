from fastapi import FastAPI, WebSocket, HTTPException
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
import time

app = FastAPI()

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands_ref = mp_hands.Hands(max_num_hands=1)

# Variables globales
landmark_8_trajectory = []   # [{time, x, y}, ...]
reference_landmarks = []     # [{frame_index: landmarks}, ...] (opcional para scatter)

def process_reference_video(video_path: str):
    """Procesa el video de referencia y extrae trayectoria del landmark 8."""
    global landmark_8_trajectory, reference_landmarks
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir {video_path}")

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_ref.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer landmark 8
                lm8 = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(lm8.x * w), int(lm8.y * h)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # segundos
                landmark_8_trajectory.append({'time': timestamp, 'x': cx, 'y': cy})

                # Opcional: guardar todos los landmarks del frame
                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                reference_landmarks.append(landmarks)
        else:
            # Si no hay mano, agrega vacío
            reference_landmarks.append([])

    cap.release()
    hands_ref.close()
    return landmark_8_trajectory

@app.on_event("startup")
async def startup():
    try:
        process_reference_video("videos/referencia.mp4")
    except FileNotFoundError:
        print("⚠️  Video de referencia no encontrado. El sistema funcionará sin datos de referencia.")
        landmark_8_trajectory.clear()
        reference_landmarks.clear()

@app.get("/reference-trajectory")
async def get_reference_trajectory():
    return {"landmark_8_trajectory": landmark_8_trajectory}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Crear una instancia de Hands para esta sesión
    hands_user = mp_hands.Hands(max_num_hands=1)
    start_time = time.time()

    try:
        while True:
            # Recibir bytes (frame JPEG)
            img_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Imagen no válida"})
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_user.process(image_rgb)

            response = {"landmarks": [], "fingertip": None}

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Todos los landmarks
                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                response["landmarks"] = landmarks

                # Punta del índice (landmark 8)
                lm8 = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                cx, cy = int(lm8.x * w), int(lm8.y * h)
                response["fingertip"] = {"x": cx, "y": cy}

            await websocket.send_json(response)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        hands_user.close()