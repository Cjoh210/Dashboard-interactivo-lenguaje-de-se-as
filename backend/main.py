from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import cv2
import base64
import mediapipe as mp
import json
from pathlib import Path
import numpy as np

app = FastAPI()

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Endpoint WebSocket /ws
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # Procesar el prefijo base64 (si lo hay)
            if data.startswith('data:image/jpeg;base64,'):
                data = data.split(',')[1]

            # Decodificar la imagen base64
            img_bytes = base64.b64decode(data)

            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image data")

            # Procesar la imagen con MediaPipe Hands
            results = hands.process(image)

            landmarks_data = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                    landmarks_data.append(landmarks)
            else:
                landmarks_data.append([])

            await websocket.send_json(landmarks_data)

    except Exception as e:
        print(f"Error processing frame: {e}")
        await websocket.close()

# Endpoint HTTP simple para verificar que el servidor está vivo
@app.get("/ping")
async def ping():
    return {"message": "pong"}

# Servir archivos estáticos desde la carpeta 'videos'
@app.get("/videos/{file_path:path}")
async def serve_static_file(file_path: str):
    file_path = Path("videos") / file_path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

# Endpoint GET para obtener los landmarks del video 'referencia.mp4'
@app.get("/reference-landmarks")
async def get_reference_landmarks():
    reference_video = Path("videos/referencia.mp4")
    if not reference_video.exists() or not reference_video.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")

    landmarks_list = []

    cap = cv2.VideoCapture(str(reference_video))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)
        else:
            landmarks_list.append([])

    cap.release()
    
    return {"landmarks": landmarks_list}

# Iniciar la aplicación
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)