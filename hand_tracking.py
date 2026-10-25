import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path("models") / "hand_landmarker.task"

# Líneas que conectan las articulaciones de la mano.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # Meñique
    (0, 17)
]


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


if not MODEL_PATH.exists():
    print(f"No se encontró el modelo: {MODEL_PATH}")
    raise SystemExit

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("No se pudo abrir la cámara.")
    raise SystemExit

start_time = time.perf_counter()

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("No se pudo obtener la imagen.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            for hand_landmarks in result.hand_landmarks:
                draw_hand(frame, hand_landmarks)

            cv2.putText(
                frame,
                "Presiona Q para cerrar",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("ASL Visual Coach - Deteccion de manos", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()