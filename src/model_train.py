"""
Step 3 - Model Training
========================
Trains an XGBoost classifier on the extracted landmarks.
Also benchmarks SVM and Random Forest for comparison.
Saves the best model and label mapping to the `models/` directory.

Usage:
    python src/model_train.py
"""

import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'models', 'landmarks_data.pkl')
MODEL_OUT   = os.path.join(BASE_DIR, 'models', 'hand_gesture_model.pkl')
MAPPING_OUT = os.path.join(BASE_DIR, 'models', 'label_mapping.json')
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load data ─────────────────────────────────────────────────────────────
    with open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)

    X = np.array([item['landmarks'].flatten() for item in dataset])
    y_raw = [item['label'] for item in dataset]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── XGBoost (primary model) ───────────────────────────────────────────────
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    xgb_acc = xgb_model.score(X_test, y_test)
    print(f'XGBoost  Accuracy : {xgb_acc:.4f}')

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_acc = svm_model.score(X_test, y_test)
    print(f'SVM      Accuracy : {svm_acc:.4f}')

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = rf_model.score(X_test, y_test)
    print(f'Random Forest Accuracy : {rf_acc:.4f}')

    # ── Save best model (XGBoost) and label mapping ───────────────────────────
    label_mapping = {str(idx): label for idx, label in enumerate(le.classes_)}
    with open(MAPPING_OUT, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(xgb_model, f)

    print(f"\nModel saved  → {MODEL_OUT}")
    print(f"Label map    → {MAPPING_OUT}")
