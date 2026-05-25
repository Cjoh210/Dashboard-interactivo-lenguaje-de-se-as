# Hand Landmarks API

API REST + WebSocket para detección de landmarks de manos en tiempo real, construida con **FastAPI** y **MediaPipe**.

---

## Estructura del proyecto

```
hand_landmarks_api/
├── main.py               # Aplicación principal
├── requirements.txt      # Dependencias Python
├── Dockerfile
├── docker-compose.yml
├── videos/               # Coloca aquí referencia.mp4 y otros videos
└── tests/
    ├── test_api.py        # Tests unitarios e integración (pytest)
    ├── ws_client_test.py  # Cliente WebSocket de prueba (Python)
    └── client.html        # Cliente WebSocket de prueba (navegador)
```

---

## Instalación local

### 1. Requisitos previos

- Python 3.10 o superior
- `pip`
- (Opcional) Cámara web para el cliente de prueba

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **macOS / Linux:** si OpenCV no encuentra librerías de sistema, instala:
> ```bash
> # Ubuntu/Debian
> sudo apt install libgl1 libglib2.0-0
> # macOS (Homebrew)
> brew install opencv
> ```

### 3. Iniciar el servidor

```bash
python main.py
# o con recarga automática:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

El servidor queda disponible en `http://localhost:8000`.

---

## Endpoints

| Método | Ruta                     | Descripción                                         |
|--------|--------------------------|-----------------------------------------------------|
| `GET`  | `/ping`                  | Health check                                        |
| `GET`  | `/docs`                  | Swagger UI (documentación interactiva)              |
| `GET`  | `/redoc`                 | ReDoc (documentación alternativa)                   |
| `WS`   | `/ws`                    | Stream de landmarks en tiempo real                  |
| `GET`  | `/videos/{path}`         | Servir archivos de video estáticos                  |
| `GET`  | `/reference-landmarks`   | Extraer landmarks del video `videos/referencia.mp4` |

---

## Uso del WebSocket `/ws`

El cliente envía frames como **string base64** (con o sin prefijo `data:image/jpeg;base64,`) y recibe un array JSON de landmarks.

**Formato de respuesta:**
```json
[
  [
    { "x": 0.52, "y": 0.34, "z": -0.01 },
    ...
  ]
]
```
- Un array por mano detectada.
- `[[]]` si no se detecta ninguna mano.
- `{"error": "mensaje"}` si el frame no es válido.

Cada mano tiene **21 landmarks** en coordenadas normalizadas (0–1).

---

## Video de referencia

Coloca el archivo `referencia.mp4` en la carpeta `videos/`:

```
videos/
└── referencia.mp4
```

Luego accede a:
```
GET http://localhost:8000/reference-landmarks
```

---

## Pruebas

### Tests automatizados

```bash
pytest tests/test_api.py -v
```

### Cliente Python (requiere cámara)

```bash
# Con cámara:
python tests/ws_client_test.py

# Sin cámara (frame en blanco):
python tests/ws_client_test.py --blank
```

### Cliente navegador

Abre `tests/client.html` directamente en el navegador (Chrome/Firefox).  
Permite ver el feed de la cámara con los landmarks dibujados en tiempo real.

---

## Docker

```bash
# Construir y levantar
docker-compose up --build

# En background
docker-compose up -d
```

---

## Notas importantes

- **RGB vs BGR:** MediaPipe requiere imágenes en RGB. El código convierte automáticamente desde BGR (formato de OpenCV).
- **Hilo separado:** el procesado de imágenes y video es CPU-bound; se ejecuta en `asyncio.to_thread()` para no bloquear el event loop.
- **Path traversal:** el endpoint `/videos/` valida que la ruta no salga del directorio `videos/`.
- **CORS:** está abierto a `*` por defecto. Ajusta `allow_origins` en `main.py` para producción.
