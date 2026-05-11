<div align="center">

# 🤟 Sign Language Recognition

Real-time hand gesture recognition system using **MediaPipe** + **XGBoost**.  
Point your hand at the webcam — the model reads your signs and builds words live.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## 📋 Table of Contents

- [Demo](#-demo)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Supported Gestures](#-supported-gestures)
- [Getting Started](#-getting-started)
- [Pipeline — Step by Step](#-pipeline--step-by-step)
- [Controls](#-controls-during-real-time-inference)
- [Model Performance](#-model-performance)
- [Extending the Dataset](#-extending-the-dataset)
- [Dependencies](#-dependencies)

---

## 🎬 Demo

> *Run `python src/deploy_real.py` and show your hand to the camera.*

The recognised gesture is appended to an on-screen word every 5 seconds.

---

## 🧠 How It Works

```
Webcam Frame
    │
    ▼
MediaPipe Hand Landmarker
    │  detects 21 3-D keypoints per hand
    ▼
Feature Engineering
    │  coordinates made relative to the wrist (landmark 0)
    │  → 63 features (21 × xyz)
    ▼
XGBoost Classifier
    │  trained on 6 000+ labelled samples
    ▼
Label → Gesture Name
```

1. **MediaPipe** detects 21 hand landmarks (x, y, z) in every frame.  
2. All coordinates are **normalised relative to the wrist** so the model is invariant to hand position and scale.  
3. A **flat 63-feature vector** is fed to an XGBoost classifier trained from scratch on your own data.  
4. Predictions are accumulated into a growing word displayed on screen.

---

## 📁 Project Structure

```
sign_language_recognition/
│
├── src/
│   ├── create_sign_lang_data.py   # Step 1 – capture training images via webcam
│   ├── create_the_input.py        # Step 2 – extract MediaPipe landmarks → .pkl
│   ├── model_train.py             # Step 3 – train & compare classifiers
│   └── deploy_real.py             # Step 4 – real-time inference
│
├── models/
│   ├── hand_landmarker.task       # Pre-trained MediaPipe hand landmarker
│   ├── hand_gesture_model.pkl     # Trained XGBoost classifier
│   ├── landmarks_data.pkl         # Extracted landmark dataset
│   └── label_mapping.json         # Index → gesture name mapping
│
├── data/                          # Raw captured images (gitignored)
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🤙 Supported Gestures

| Model Index | Gesture |
|:-----------:|:-------:|
| 0 | a |
| 1 | b |
| 2 | cool |
| 3 | d |
| 4 | done |
| 5 | e |
| 7 | l |
| 8 | n |
| 9 | o |

> You can add more gestures — see [Extending the Dataset](#-extending-the-dataset).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or newer
- A working webcam

### 1 — Clone the repo

```bash
git clone https://github.com/<your-username>/sign_language_recognition.git
cd sign_language_recognition
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run real-time inference (pre-trained model included)

```bash
python src/deploy_real.py
```

---

## 🔄 Pipeline — Step by Step

Use these steps only if you want to **collect new data and retrain** the model.

### Step 1 — Collect raw images

```bash
python src/create_sign_lang_data.py
```

- Opens your webcam and captures **1 000 images per gesture class**.
- Press **Q** to start capturing each class.
- Images are saved under `data/<class_index>/`.

### Step 2 — Extract hand landmarks

```bash
python src/create_the_input.py
```

- Runs the MediaPipe Hand Landmarker on every image.
- Saves a `landmarks_data.pkl` file to `models/`.

### Step 3 — Train the classifier

```bash
python src/model_train.py
```

- Trains **XGBoost**, **SVM**, and **Random Forest** and prints accuracy for each.
- Saves the XGBoost model to `models/hand_gesture_model.pkl`.
- Saves the updated `models/label_mapping.json`.

### Step 4 — Run real-time inference

```bash
python src/deploy_real.py
```

---

## ⌨️ Controls During Real-Time Inference

| Key | Action |
|:---:|--------|
| `Q` | Quit the application |
| `C` | Clear the accumulated word |
| `D` | Delete the last character |

---

## 📊 Model Performance

Tested on a held-out 20 % split of ~6 000 samples:

| Model | Accuracy |
|-------|----------|
| XGBoost *(saved)* | **≈ 99 %** |
| SVM (linear) | high |
| Random Forest | high |

> Results will vary depending on lighting conditions and your data quality.

---

## ➕ Extending the Dataset

1. Edit `NUMBER_OF_CLASSES` in `src/create_sign_lang_data.py` to add more classes.
2. Re-run the full 4-step pipeline.
3. Update `models/label_mapping.json` with the new gesture names (this is done automatically by `model_train.py`).

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture & image I/O |
| `mediapipe` | Hand landmark detection |
| `numpy` | Numerical operations |
| `xgboost` | Primary classifier |
| `scikit-learn` | SVM, Random Forest, train/test split |
| `matplotlib` | Landmark visualisation during feature extraction |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
Made with ❤️ using MediaPipe & XGBoost
</div>
