"""
Step 2 - Feature Extraction
============================
Processes collected images using MediaPipe Hand Landmarker.
Extracts 3D hand landmarks (relative to wrist) and saves them as a pickle dataset.

Usage:
    python src/create_the_input.py
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')
OUTPUT_PATH = os.path.join(BASE_DIR, 'models', 'landmarks_data.pkl')
# ───────────────────────────────────────────────────────────────────────────────

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand skeleton and landmark dots on a copy of the image."""
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        for connection in HAND_CONNECTIONS:
            start = hand_landmarks[connection[0]]
            end   = hand_landmarks[connection[1]]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w),   int(end.y * h)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (88, 205, 54), 2)

        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(annotated_image, (x, y), 2, (0, 0, 255), -1)

        if detection_result.handedness:
            handedness = detection_result.handedness[idx][0].category_name
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            text_x = int(min(x_coords) * w)
            text_y = int(min(y_coords) * h) - 10
            cv2.putText(annotated_image, handedness, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (88, 205, 54), 1, cv2.LINE_AA)

    return annotated_image


if __name__ == "__main__":
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )

    dataset = []

    with HandLandmarker.create_from_options(options) as landmarker:

        # ── Preview first image ────────────────────────────────────────────────
        first_label = sorted(os.listdir(DATA_DIR))[0]
        test_path = os.path.join(DATA_DIR, first_label, '0.jpg')
        if os.path.exists(test_path):
            mp_image = mp.Image.create_from_file(test_path)
            detection_result = landmarker.detect(mp_image)
            annotated = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated)
            plt.axis('off')
            plt.show()

        # ── Extract landmarks from all images ─────────────────────────────────
        for label in sorted(os.listdir(DATA_DIR)):
            label_dir = os.path.join(DATA_DIR, label)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                try:
                    mp_image = mp.Image.create_from_file(img_path)
                    result = landmarker.detect(mp_image)

                    if result.hand_landmarks:
                        base_x = result.hand_landmarks[0][0].x
                        base_y = result.hand_landmarks[0][0].y
                        base_z = result.hand_landmarks[0][0].z

                        landmarks = []
                        for lm in result.hand_landmarks[0]:
                            landmarks.append([lm.x - base_x,
                                              lm.y - base_y,
                                              lm.z - base_z])

                        dataset.append({
                            'landmarks': np.array(landmarks).flatten(),
                            'label': label
                        })
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(dataset)} samples to {OUTPUT_PATH}")
