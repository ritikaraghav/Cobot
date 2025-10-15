import os
import absl.logging
import cv2
import numpy as np
import json
import pyttsx3
import requests
import threading
import time
import speech_recognition as sr
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque

# ---------------------- Suppress Logs ----------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

# ---------------------- ESP32/ESP8266 Gripper Control ----------------------
ESP_IP = "192.168.4.1"  # Change to your ESP IP
ESP_TIMEOUT = 3
ESP_RETRY = 1

def grip_close():
    url = f"http://{ESP_IP}/grip?action=open"
    for attempt in range(ESP_RETRY + 1):
        try:
            requests.get(url, timeout=ESP_TIMEOUT)
            print("Gripper opened")
            return True
        except Exception as e:
            print(f"Failed to open gripper (attempt {attempt+1}): {e}")
            time.sleep(0.5)
    return False

def grip_open():
    url = f"http://{ESP_IP}/grip?action=close"
    for attempt in range(ESP_RETRY + 1):
        try:
            requests.get(url, timeout=ESP_TIMEOUT)
            print("Gripper closed")
            return True
        except Exception as e:
            print(f"Failed to close gripper (attempt {attempt+1}): {e}")
        
    return False

# ---------------------- TTS ----------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# ---------------------- Load CNN Model + Labels ----------------------
MODEL_PATH = "/Users/ritika/cobot/gesture_recognition/my_gestures_mobilenet.keras"
try:
    classifier = load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    raise SystemExit(1)

class_labels = {0: "Straight Dissection clamp"}

label_lookup = {v.lower(): v for v in class_labels.values()}

# ---------------------- MediaPipe Hands ----------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# ---------------------- Webcam Setup ----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------------- State Management ----------------------
state = "idle"
current_tool = None
hold_time = 5

state_lock = threading.Lock()
action_in_progress = threading.Event()

# ---------------------- Prediction Buffer ----------------------
pred_buffer = deque(maxlen=5)

# ---------------------- Tool Actions ----------------------
def pick_tool(tool):
    global state, current_tool
    if action_in_progress.is_set():
        return
    action_in_progress.set()
    try:
        with state_lock:
            state = "picking"
            current_tool = tool
        speak(f"{tool} identified")
        grip_open()
        time.sleep(0.7)
        grip_close()
        speak(f"Picking {tool}")
        with state_lock:
            state = "holding"
        start_hold = time.time()
        while time.time() - start_hold < hold_time:
            time.sleep(0.5)
        with state_lock:
            state = "ready_to_release"
    finally:
        action_in_progress.clear()

def release_tool(tool):
    global state, current_tool
    if action_in_progress.is_set():
        return
    action_in_progress.set()
    try:
        with state_lock:
            if current_tool is not None:
                speak(f"Releasing {tool}")
                grip_open()
            state = "idle"
            current_tool = None
    finally:
        action_in_progress.clear()

def process_tool_command(tool_name):
    global state, current_tool
    tool_norm = tool_name
    if tool_norm is None:
        return
    with state_lock:
        cur_state = state
        cur_tool = current_tool
    if cur_state == "idle":
        threading.Thread(target=pick_tool, args=(tool_norm,), daemon=True).start()
        return
    if cur_state == "ready_to_release" and cur_tool == tool_norm:
        speak(f"Releasing Straight Dissection clamp")
        threading.Thread(target=release_tool, args=(tool_norm,), daemon=True).start()
        return

# ---------------------- Voice Recognition ----------------------
recognizer = sr.Recognizer()
MIC_INDEX = None
mic = sr.Microphone(device_index=MIC_INDEX) if MIC_INDEX is not None else sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)

def voice_callback(recognizer, audio):
    try:
        command = recognizer.recognize_google(audio).lower()
        print("[VOICE] Heard:", command)
        for label_lower, proper_name in label_lookup.items():
            if label_lower in command:
                process_tool_command(proper_name)
    except sr.UnknownValueError:
        print("[VOICE] Could not understand")
    except sr.RequestError as e:
        print("[VOICE] Request error:", e)

stop_listening = recognizer.listen_in_background(mic, voice_callback)

# ---------------------- Main Loop ----------------------
hand_detected = False  # to prevent repeated TTS

print("Press Enter in this terminal to exit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        label = "No Gesture"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not hand_detected:
                hand_detected = True
                label = "Straight Dissection clamp"
                process_tool_command(label)
                speak(label)
        else:
            hand_detected = False

        with state_lock:
            st = state
            ct = current_tool
        cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {st}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if ct:
            cv2.putText(frame, f"Tool: {ct}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Gesture + Voice Control", frame)
        key = cv2.waitKey(1)
        if key == 13:  # Enter
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    stop_listening(wait_for_stop=False)
    print("Program terminated cleanly.")
