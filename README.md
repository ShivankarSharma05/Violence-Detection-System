# 🚌 AI Violence Detection & Emergency Alert System

An intelligent computer vision system designed for real-time violence detection and emergency response (tailored for public transport/buses). Built with **TensorFlow / Keras (MobileNetV2)**, **OpenCV**, **Pygame**, and **Flask**.

---

## 📌 Features

- 🎥 **Frame Extraction Tool**: Converts raw video datasets into resized image frames for training.
- 🧠 **Transfer Learning Model**: Employs MobileNetV2 pre-trained on ImageNet to classify binary states (`VIOLENCE` vs `NORMAL`).
- ⚡ **Real-Time Detection**: Performs dynamic frame buffer smoothing (5-frame moving window) on webcam feed for robust detection.
- 🚨 **Audible Alarm System**: Plays an alarm sound (`alarm.mp3`) via Pygame whenever violence is detected.
- 🆘 **Emergency SOS Dispatch**: Key-triggered (`S` key) SOS alert mechanism with built-in cooldown logic.
- 🌐 **Flask REST API**: Exposes an API endpoint (`/run-ai`) to initiate detection remotely.
- ⚡ **GPU Verification**: Includes a script to test TensorFlow GPU matrix multiplication and acceleration.

---

## 🛠️ Project Architecture & Pipeline

```text
Camera / Video Stream
       │
       ▼
Frame Preprocessing (224x224, Normalized)
       │
       ▼
MobileNetV2 Feature Extraction & Binary Classification
       │
       ▼
Moving Average Prediction (Buffer Size = 5)
       ├───> Normal (Prediction <= 0.5) ───> Stop Alarm
       └───> Violence (Prediction > 0.5) ───> Play Alarm & Trigger Alert
                                                      │
                                                      ▼
                                           User Presses 'S' ───> Dispatch SOS
```

---

## 📂 Repository Structure

```text
MODEL_bus/
├── app.py               # Flask REST API to run detection script remotely
├── detect.py            # Real-time webcam violence detection & live display
├── train_model.py       # MobileNetV2 model training pipeline
├── extract_frames.py    # Extracts training frames from raw videos
├── alarm_sound.py       # Pygame audio manager for alarm playback
├── sos.py               # SOS notification dispatcher with cooldown
├── testALARM.py         # Utility script to test alarm audio playback
├── nano check.py        # GPU benchmark & memory growth verification script
├── alarm.mp3            # Alarm audio track
├── flowchart.txt        # High-level architecture flowchart summary
├── .gitignore           # Ignored files (models, datasets, venv, cache)
└── requirements.txt     # Python dependencies list
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- Webcam (for live detection)
- CUDA-compatible GPU (optional, for accelerated inference & training)

### 2. Environment Setup

Clone the repository and create a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/macOS)
source venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage Instructions

### 1. Extract Dataset Frames (Optional)
Place raw `.mp4` videos into `videos/violence` and `videos/non_violence`, then run:
```bash
python extract_frames.py
```

### 2. Train the Detection Model
Train MobileNetV2 on the extracted frame dataset:
```bash
python train_model.py
```
*The best model weights will be saved to `models/violence_model.h5`.*

### 3. Run Real-Time Violence Detection
Start live camera monitoring:
```bash
python detect.py
```
- **Controls**:
  - Press **`S`** when an alert is active to send an SOS.
  - Press **`Q`** to exit detection mode.

### 4. Run via Flask REST API
Start the API server:
```bash
python app.py
```
- Send a GET request to `http://127.0.0.1:5000/run-ai` to trigger `detect.py`.

### 5. Check GPU Acceleration
Verify TensorFlow GPU availability and benchmark matrix multiplication speed:
```bash
python "nano check.py"
```

---

## 📄 License
This project is for educational and research purposes.
