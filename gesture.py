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

# ---------------------- Suppress Logs ----------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

# ---------------------- ESP32/ESP8266 Gripper Control ----------------------
ESP_IP = "192.168.4.1"  # Change to your ESP IP (AP default)
ESP_TIMEOUT = 3
ESP_RETRY = 1  # number of retries for HTTP requests

def grip_open():
    """Command the ESP gripper to OPEN (named correctly)."""
    url = f"http://{ESP_IP}/grip?action=open"
    for attempt in range(ESP_RETRY + 1):
        try:
            requests.get(url, timeout=ESP_TIMEOUT)
            print("Gripper opened")
            return True
        except Exception as e:
            print(f"Failed to open gripper (attempt {attempt+1}): {e}")
            time.sleep(0.2)
    return False

def grip_close():
    """Command the ESP gripper to CLOSE (named correctly)."""
    url = f"http://{ESP_IP}/grip?action=close"
    for attempt in range(ESP_RETRY + 1):
        try:
            requests.get(url, timeout=ESP_TIMEOUT)
            print("Gripper closed")
            return True
        except Exception as e:
            print(f"Failed to close gripper (attempt {attempt+1}): {e}")
            time.sleep(0.2)
    return False

# ---------------------- TTS ----------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text):
    """Run TTS synchronously but called inside worker threads to avoid blocking main loop."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# ---------------------- Load Model + Labels ----------------------
try:
    classifier = load_model("my_gestures_mobilenet.keras")
except Exception as e:
    print("Error loading model:", e)
    raise SystemExit(1)

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

# Make a lowercase lookup for simple matching
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
state = "idle"  # idle, picking, holding, ready_to_release
current_tool = None
hold_time = 5  # seconds to hold after picking
prediction_cooldown = 0.3
last_predict_time = time.time()
last_activity_time = time.time()
gesture_timeout = 7  # seconds without activity -> voice listening

# Threading primitives
state_lock = threading.Lock()
action_in_progress = threading.Event()  # indicates a pick/release in progress (prevents duplicates)

# ---------------------- Tool Actions ----------------------
def pick_tool(tool):
    """Threaded function to pick a tool (opens then closes gripper)."""
    global state, current_tool

    # Avoid multiple simultaneous picks/releases
    if action_in_progress.is_set():
        print(f"pick_tool({tool}) requested but action already in progress.")
        return

    action_in_progress.set()
    try:
        with state_lock:
            state = "picking"
            current_tool = tool
        print(f"[ACTION] {tool} identified -> starting pick sequence")

        # Announce
        speak(f"{tool} identified")

        # Open gripper then close to pick
        ok = grip_open()
        time.sleep(0.5)
        ok2 = grip_close()
        speak(f"Picking {tool}")
        print(f"[ACTION] Picking {tool} (open ok: {ok}, close ok: {ok2})")

        with state_lock:
            state = "holding"

        # Hold for configured time (can be interrupted by other logic)
        start_hold = time.time()
        while time.time() - start_hold < hold_time:
            time.sleep(0.1)

        with state_lock:
            state = "ready_to_release"
        print(f"[STATE] ready_to_release (tool: {current_tool})")

    except Exception as e:
        print("Error in pick_tool:", e)
        with state_lock:
            state = "idle"
            current_tool = None
    finally:
        action_in_progress.clear()

def release_tool(tool):
    """Threaded function to release a tool (opens gripper)."""
    global state, current_tool

    if action_in_progress.is_set():
        print(f"release_tool({tool}) requested but action already in progress.")
        return

    action_in_progress.set()
    try:
        with state_lock:
            # Only release if the tool matches current_tool to avoid accidental releases
            if current_tool is None:
                print("release_tool called but no current_tool set.")
            else:
                print(f"[ACTION] Releasing {tool}")
                speak(f"Releasing {tool}")
                ok = grip_open()
                print(f"[ACTION] release done (open ok: {ok})")

            # Reset state
            with state_lock:
                state = "idle"
                current_tool = None
    except Exception as e:
        print("Error in release_tool:", e)
        with state_lock:
            state = "idle"
            current_tool = None
    finally:
        action_in_progress.clear()

def release_and_pick(new_tool):
    """Release current tool (if any) then pick the new tool."""
    global current_tool
    print(f"[TRANSITION] release_and_pick: switching to {new_tool}")
    # Release existing tool if present
    with state_lock:
        old_tool = current_tool
    if old_tool is not None:
        release_tool(old_tool)
        time.sleep(0.4)
    # Now pick new
    pick_tool(new_tool)

def process_tool_command(tool_name):
    """Unified entry for both gesture and voice commands."""
    global state, current_tool

    # normalize
    tool_norm = tool_name
    if tool_norm is None:
        return

    with state_lock:
        cur_state = state
        cur_tool = current_tool

    # If idle -> pick
    if cur_state == "idle":
        print(f"[CMD] idle -> pick {tool_norm}")
        threading.Thread(target=pick_tool, args=(tool_norm,), daemon=True).start()
        return

    # If ready_to_release and same tool -> release
    if cur_state == "ready_to_release" and cur_tool == tool_norm:
        print(f"[CMD] ready_to_release and tool matches -> release {tool_norm}")
        threading.Thread(target=release_tool, args=(tool_norm,), daemon=True).start()
        return

    # If different tool while busy -> interrupt: release current then pick new
    if cur_tool is not None and cur_tool != tool_norm:
        print(f"[CMD] Request to switch tools: {cur_tool} -> {tool_norm}")
        # start a background transition if not already busy
        if not action_in_progress.is_set():
            threading.Thread(target=release_and_pick, args=(tool_norm,), daemon=True).start()
        else:
            print("[CMD] action in progress, ignoring switch request for now.")
        return

    # If same tool requested while picking/holding -> ignore duplicate
    print(f"Cannot process {tool_norm}. Current state: {cur_state}, Current tool: {cur_tool}")

# ---------------------- Voice Recognition ----------------------
recognizer = sr.Recognizer()
# Optionally set a specific device index if your machine has multiple mics
# Uncomment and set device_index after listing with sr.Microphone.list_microphone_names()
mic = sr.Microphone()

def listen_for_command_once():
    """Listen once and process command (blocking for a short time). Intended to be called repeatedly in a dedicated thread."""
    global last_activity_time
    try:
        with mic as source:
            print("[VOICE] Listening for voice command...")
            # Adjust for ambient noise quickly
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
        # Try Google (requires internet)
        try:
            command = recognizer.recognize_google(audio).lower()
            print("[VOICE] Heard:", command)
        except sr.UnknownValueError:
            print("[VOICE] Could not understand voice")
            return
        except sr.RequestError as e:
            print("[VOICE] Could not request results from speech service; {0}".format(e))
            return

        # Find matching tool
        for label_lower, proper_name in label_lookup.items():
            if label_lower in command:
                print(f"[VOICE] Matched tool {proper_name} from '{command}'")
                process_tool_command(proper_name)
                last_activity_time = time.time()
                return

        print("[VOICE] No valid tool found in command:", command)

    except sr.WaitTimeoutError:
        # no speech within timeout
        # print("[VOICE] No speech detected (timeout)")
        return
    except Exception as e:
        print("[VOICE] Unexpected voice error:", e)
        return

def voice_listener_loop():
    """Persistent voice thread - calls listen_for_command_once() repeatedly with short sleeps."""
    print("[VOICE] Voice listener thread started")
    while True:
        # Only actively listen when system is idle or ready_to_release
        with state_lock:
            allow_listen = (state in ("idle", "ready_to_release"))
        if allow_listen:
            listen_for_command_once()
        time.sleep(0.2)

# Start the persistent voice thread
voice_thread = threading.Thread(target=voice_listener_loop, daemon=True)
voice_thread.start()

# ---------------------- Main Loop ----------------------
print("Press Enter in this terminal to exit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        label = "No Gesture"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prediction cooldown to save CPU
            if time.time() - last_predict_time > prediction_cooldown:
                last_predict_time = time.time()
                # Only process the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x1, y1 = int(min(x_coords) * w) - 20, int(min(y_coords) * h) - 20
                x2, y2 = int(max(x_coords) * w) + 20, int(max(y_coords) * h) + 20

                roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, (128, 128))
                    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    roi_input = roi_rgb.astype("float32") / 255.0
                    roi_input = np.expand_dims(roi_input, axis=0)

                    try:
                        preds = classifier.predict(roi_input, verbose=0)
                        result = int(np.argmax(preds))
                        max_conf = float(np.max(preds))
                        if max_conf > 0.7:
                            label = class_labels.get(result, "Unknown")
                            if label != "Unknown":
                                last_activity_time = time.time()
                                process_tool_command(label)
                    except Exception as e:
                        # model/prediction error
                        # print("Prediction error:", e)
                        label = "No Gesture"

        # If no activity for a while, let voice thread pick up (voice thread already checks state)
        # We only update last_activity_time on valid detections / voice matches (done above)

        # UI overlay for debugging
        with state_lock:
            st = state
            ct = current_tool
        cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {st}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if ct:
            cv2.putText(frame, f"Tool: {ct}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Gesture + Voice Control", frame)

        # Exit on Enter key
        key = cv2.waitKey(1)
        if key == 13:  # Enter
            break

except KeyboardInterrupt:
    print("Interrupted by user")

except Exception as e:
    print("Unexpected error in main:", e)

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Program terminated cleanly.")
