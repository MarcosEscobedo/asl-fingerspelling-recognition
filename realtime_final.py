import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import pandas as pd


HAND_MODEL_PATH = Path("models") / "hand_landmarker.task"
CLASSIFIER_PATH = Path("models") / "asl_classifier_final.joblib"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize_landmarks(landmarks):
    wrist = landmarks[0]

    coordinates = []

    for landmark in landmarks:
        coordinates.append([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
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
            z / scale,
        ])

    return normalized


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2
        )

    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )


classifier = joblib.load(CLASSIFIER_PATH)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("No se pudo abrir la cámara.")
    raise SystemExit


prediction_history = deque(maxlen=7)

start_time = time.perf_counter()


try:

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

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

            displayed_prediction = "-"
            displayed_confidence = 0

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                draw_hand(frame, landmarks)

                features = normalize_landmarks(landmarks)

                input_data = pd.DataFrame(
                    [features],
                    columns=classifier.feature_names_in_
                )

                prediction = classifier.predict(
                    input_data
                )[0]

                probabilities = classifier.predict_proba(
                    input_data
                )[0]

                confidence = max(probabilities)

                prediction_history.append(prediction)

                if len(prediction_history) == 7:

                    counts = Counter(prediction_history)

                    stable_prediction, frequency = (
                        counts.most_common(1)[0]
                    )

                    if frequency >= 5:
                        displayed_prediction = stable_prediction
                        displayed_confidence = confidence

            else:
                prediction_history.clear()

            cv2.putText(
                frame,
                f"ASL: {displayed_prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )

            if displayed_prediction != "-":
                cv2.putText(
                    frame,
                    f"Confidence: {displayed_confidence:.0%}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "Static fingerspelling | J/Z not supported",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "Q = Exit",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "AI-ASL Fingerspelling Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


finally:

    camera.release()
    cv2.destroyAllWindows()