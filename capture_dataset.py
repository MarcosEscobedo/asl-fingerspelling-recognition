import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path("models") / "hand_landmarker.task"
DATA_PATH = Path("data") / "asl_static.csv"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
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


def normalize_landmarks(landmarks):
    wrist = landmarks[0]

    coordinates = []

    for landmark in landmarks:
        coordinates.append([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])

    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in coordinates
    )

    if scale == 0:
        scale = 1

    normalized = []

    for x, y, z in coordinates:
        normalized.extend([
            x / scale,
            y / scale,
            z / scale
        ])

    return normalized


def save_sample(label, participant, session, handedness, landmarks):

    DATA_PATH.parent.mkdir(exist_ok=True)

    file_exists = DATA_PATH.exists()

    row = [
        label,
        participant,
        session,
        handedness
    ] + normalize_landmarks(landmarks)

    with DATA_PATH.open("a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists:

            header = [
                "label",
                "participant",
                "session",
                "handedness"
            ]

            for point in range(21):
                header.extend([
                    f"x{point}",
                    f"y{point}",
                    f"z{point}"
                ])

            writer.writerow(header)

        writer.writerow(row)


if len(sys.argv) < 4:
    print("Uso:")
    print("python capture_dataset.py LETRA PARTICIPANTE SESION")
    print()
    print("Ejemplo:")
    print("python capture_dataset.py A P01 S01")
    raise SystemExit


label = sys.argv[1].upper()
participant = sys.argv[2].upper()
session = sys.argv[3].upper()


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


camera = cv2.VideoCapture(0)

start_time = time.perf_counter()
saved_samples = 0


try:

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

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

            detected_landmarks = None
            handedness = "Unknown"

            if result.hand_landmarks:

                detected_landmarks = result.hand_landmarks[0]

                draw_hand(
                    frame,
                    detected_landmarks
                )

                if result.handedness:
                    handedness = result.handedness[0][0].category_name


            cv2.putText(
                frame,
                f"{label} | {participant} | {session}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Muestras: {saved_samples}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "S = guardar | Q = cerrar",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ASL Dataset Capture",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):

                if detected_landmarks is not None:

                    save_sample(
                        label,
                        participant,
                        session,
                        handedness,
                        detected_landmarks
                    )

                    saved_samples += 1

                    print(
                        f"{label} - muestra {saved_samples}"
                    )

            elif key == ord("q"):
                break


finally:

    camera.release()
    cv2.destroyAllWindows()


print(f"Muestras guardadas: {saved_samples}")