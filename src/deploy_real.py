"""
Step 4 - Real-Time Inference
=============================
Runs the trained model on your webcam feed in real time.
Recognised gestures are accumulated into a word on screen.

Controls
--------
  Q  — quit
  C  — clear accumulated word
  D  — delete last character

Usage:
    python src/deploy_real.py
"""

import os
import json
import pickle
import time
import numpy as np
import cv2
import mediapipe as mp

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'hand_gesture_model.pkl')
MAPPING_PATH = os.path.join(BASE_DIR, 'models', 'label_mapping.json')
LANDMARKER   = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')
# ───────────────────────────────────────────────────────────────────────────────

PREDICTION_DELAY = 5.0   # seconds between predictions

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand skeleton overlay."""
    annotated = np.copy(rgb_image)
    h, w, _ = annotated.shape

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        for s, e in HAND_CONNECTIONS:
            x1, y1 = int(hand_landmarks[s].x * w), int(hand_landmarks[s].y * h)
            x2, y2 = int(hand_landmarks[e].x * w), int(hand_landmarks[e].y * h)
            cv2.line(annotated, (x1, y1), (x2, y2), (88, 205, 54), 2)
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)

    return annotated


# ── Load model & label mapping ────────────────────────────────────────────────
with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

model = pickle.load(open(MODEL_PATH, 'rb'))

# ── MediaPipe options ─────────────────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# ── Main loop ─────────────────────────────────────────────────────────────────
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    last_prediction_time = 0
    accumulated_word = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(image_tensor, timestamp_ms)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

            rel_landmarks = [
                [lm.x - base_x, lm.y - base_y, lm.z - base_z]
                for lm in landmarks
            ]
            landmarks_array = np.array(rel_landmarks).flatten().reshape(1, -1)

            current_time = time.time()
            if current_time - last_prediction_time >= PREDICTION_DELAY:
                prediction_key = str(model.predict(landmarks_array)[0])
                if prediction_key in label_mapping:
                    predicted_letter = label_mapping[prediction_key]
                    accumulated_word += predicted_letter
                    print(f'\rCurrent Word: {accumulated_word}          ', end='')
                last_prediction_time = current_time

        cv2.putText(frame, f'Word: {accumulated_word}',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        bgr_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Gesture Recognition', bgr_annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            accumulated_word = ""
        elif key == ord('d'):
            accumulated_word = accumulated_word[:-1]

    cap.release()
    cv2.destroyAllWindows()
