import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import pygame

from alarm_sound import play_alarm, stop_alarm
from sos import send_sos

# =========================
# LOAD MODEL
# =========================

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "violence_model.h5")

model = load_model(model_path)

IMG_SIZE = 224

cap = cv2.VideoCapture(0)

frame_buffer = []
alarm_on = False
show_alert = False

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    frame_buffer.append(img)

    # Keep last 5 frames
    if len(frame_buffer) > 5:
        frame_buffer.pop(0)

    # =========================
    # PREDICTION
    # =========================
    if len(frame_buffer) == 5:
        preds = []

        for f in frame_buffer:
            f = np.reshape(f, (1, IMG_SIZE, IMG_SIZE, 3))
            pred = model.predict(f, verbose=0)[0][0]
            preds.append(pred)

        avg_pred = sum(preds) / len(preds)

        # =========================
        # VIOLENCE LOGIC
        # =========================
        if avg_pred > 0.5:
            label = "VIOLENCE"
            color = (0, 0, 255)
            show_alert = True

            # Start alarm
            if not alarm_on:
                alarm_on = True
                play_alarm()

        else:
            label = "NORMAL"
            color = (0, 255, 0)
            show_alert = False

            # Stop alarm
            if alarm_on:
                alarm_on = False
                stop_alarm()

        # Display result
        cv2.putText(frame, f"{label} ({avg_pred:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

    # =========================
    # ALERT TEXT
    # =========================
    if show_alert:
        cv2.putText(frame, "!!! ALERT: VIOLENCE DETECTED !!!",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        cv2.putText(frame, "Press S to send SOS",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("Violence Detection System", frame)

    # =========================
    # KEY HANDLING
    # =========================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and show_alert:
        send_sos()

    if key == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()