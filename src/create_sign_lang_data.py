"""
Step 1 - Data Collection
========================
Captures images from your webcam for each gesture class.
Images are saved to the `data/` directory.

Usage:
    python src/create_sign_lang_data.py
"""

import os
import cv2

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUMBER_OF_CLASSES = 10
DATASET_SIZE = 1000      # images per class
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for j in range(NUMBER_OF_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {j}')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start  :)',
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(50) == ord('q'):
            break

    # Capture frames
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(50)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
