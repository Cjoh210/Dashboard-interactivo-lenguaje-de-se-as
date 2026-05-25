"""
Cliente de prueba WebSocket — envía frames de la cámara al servidor.
Uso:  python tests/ws_client_test.py
"""

import asyncio
import base64
import cv2
import websockets
import json


SERVER_URL = "ws://localhost:8000/ws"


async def test_with_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    print(f"✅ Conectando a {SERVER_URL} …")
    async with websockets.connect(SERVER_URL) as ws:
        print("✅ Conectado. Presiona Ctrl+C para salir.\n")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, buffer = cv2.imencode(".jpg", frame)
                b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

                await ws.send(b64)
                response = await ws.recv()
                data = json.loads(response)

                if data and data[0]:
                    print(f"✋ Mano detectada — {len(data[0])} landmarks")
                else:
                    print("🤚 Sin mano detectada")

                cv2.imshow("Test Client", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nSaliendo…")
        finally:
            cap.release()
            cv2.destroyAllWindows()


async def test_with_blank_frame():
    """Test rápido con un frame negro (sin cámara)."""
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", frame)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    print(f"✅ Conectando a {SERVER_URL} …")
    async with websockets.connect(SERVER_URL) as ws:
        await ws.send(b64)
        response = await ws.recv()
        data = json.loads(response)
        print("Respuesta del servidor:", data)
        assert data == [[]], f"Esperado [[]], recibido: {data}"
        print("✅ Test con frame en blanco OK")


if __name__ == "__main__":
    import sys

    if "--blank" in sys.argv:
        asyncio.run(test_with_blank_frame())
    else:
        asyncio.run(test_with_camera())
