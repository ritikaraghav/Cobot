import cv2
import numpy as np
import json
import os
import pyttsx3
import requests
import speech_recognition as sr
from tensorflow.keras.models import load_model
import mediapipe as mp
from typing import Any

# ----------------------
# ESP8266 CONFIG
# ----------------------
ESP_IP: str = "192.168.4.1"  # Replace with your ESP8266 IP
NO_GESTURE_FRAMES = 5  # Frames to wait before listening to voice

# Map gestures to servo positions
gesture_to_pos = {
    0: 0,     # Compass
    1: 45,    # Hemostatic Forceps
    2: 90,    # Kelly Hemostatic Forceps
    3: 135,   # Plier
    4: 180    # Spool
}

# Map voice commands to servo positions
command_to_pos = {
    "compass": 0,
    "plier": 135,
    "spool": 180,
    "hemostatic forceps": 45,
    "kelly hemostatic forceps": 90
}

# ----------------------
# Load Model
# ----------------------
classifier = load_model("my_gestures_mobilenet.keras")

if os.path.exists("class_labels.json"):
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
        class_labels = {int(k): v for k, v in class_labels.items()}
else:
    class_labels = {
        0: "Compass",
        1: "Hemostatic_Forceps",
        2: "Kelly_Hemostatic_Forceps",
        3: "Plier",
        4: "Spool"
    }
    print("⚠️ class_labels.json not found, using hardcoded labels.")

# ----------------------
# Text-to-Speech
# ----------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)
last_label = None
no_gesture_counter = 0
last_servo_pos = None  # Track last servo position

# ----------------------
# MediaPipe Hands
# ----------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# ----------------------
# Speech Recognition
# ----------------------
recognizer: Any = sr.Recognizer()
mic: sr.Microphone = sr.Microphone()

# ----------------------
# Servo Control Function
# ----------------------
def move_servo(pos: int):
    global last_servo_pos
    if pos == last_servo_pos:
        return  # don't send request if servo already at this position
    try:
        url = f"http://{ESP_IP}/servo?pos={pos}"
        response = requests.get(url, timeout=5)  # increase timeout for ESP
        print("✅ Servo moved:", response.text)
        last_servo_pos = pos
    except requests.RequestException as e:
        print("❌ Failed to move servo:", e)

# ----------------------
# Start Webcam
# ----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    label = "No Gesture Identified"
    max_confidence = 0.0
    gesture_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords) * w) - 20, int(min(y_coords) * h) - 20
            x2, y2 = int(max(x_coords) * w) + 20, int(max(y_coords) * h) + 20
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = frame[y1:y2, x1:x2]

            roi_resized = cv2.resize(roi, (128, 128))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_input = roi_rgb.astype("float32") / 255.0
            roi_input = np.expand_dims(roi_input, axis=0)

            preds = classifier.predict(roi_input, verbose=0)
            max_confidence = float(np.max(preds))
            result = int(np.argmax(preds))

            if max_confidence > 0.7:
                label = class_labels.get(result, "Unknown")
                no_gesture_counter = 0
                gesture_detected = True
                move_servo(gesture_to_pos[result])
            else:
                no_gesture_counter += 1
                if no_gesture_counter >= NO_GESTURE_FRAMES:
                    label = "No Gesture Identified"
                else:
                    label = last_label

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ----------------------
    # Voice Control if no gesture
    # ----------------------
    if not gesture_detected and no_gesture_counter >= NO_GESTURE_FRAMES:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("🎤 Say a command:")
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                print("Heard:", command)
                if command in command_to_pos:
                    move_servo(command_to_pos[command])
                    label = f"Voice: {command}"
                else:
                    label = "No Gesture / Command"
            except sr.UnknownValueError:
                label = "Could not understand audio"
            except sr.RequestError as e:
                label = f"Speech recognition error: {e}"
            except sr.WaitTimeoutError:
                label = "Waiting for voice..."

    # ----------------------
    # Speak if label changed
    # ----------------------
    if label != last_label:
        engine.say(label)
        engine.runAndWait()
        last_label = label

    # Draw label
    text = f"{label} ({max_confidence:.2f})"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Gesture + Voice Servo Control", frame)

    if cv2.waitKey(1) == 13:  # Enter to exit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
