import cv2
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

# ----------------------
# Load Model
# ----------------------
classifier = load_model("my_gestures_mobilenet.keras")

# Load class labels
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

    # ROI
    x1, y1, x2, y2 = 320, 100, 620, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    roi_resized = cv2.resize(roi, (128, 128))  # resize correctly
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # convert to RGB
    roi_input = roi_rgb.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_input, axis=0)  # add batch dimension
    
    # Prediction
    preds = classifier.predict(roi_input, verbose=0)
    result = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Draw
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    label = class_labels.get(result, "Unknown")
    text = f"{label} ({confidence:.2f})"
    cv2.putText(display_frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", display_frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()


"""
import cv2
import os
import time

# ----------------------
# Create directory if it doesn't exist
# ----------------------
def makedir(path):
    os.makedirs(path, exist_ok=True)

# ----------------------
# Parameters
# ----------------------
gestures = [
    ("Compass", "compass"),
    ("Spool", "spool"),
    ("Plier", "plier"),
    ("Hemostatic_Forceps", "hemostatic_forceps"),
    ("Kelly_Hemostatic_Forceps", "kelly_hemostatic_forceps")
]

images_per_gesture = 500   # Number of images to collect
delay_between_images = 0.2  # seconds

# ----------------------
# Start camera
# ----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

for gesture_name, folder_name in gestures:
    image_count = 0
    gesture_path = f"./handgestures/train/{folder_name}/"
    makedir(gesture_path)
    print(f"\n📸 Get ready to record gesture: {gesture_name}")
    print("Starting in 3 seconds...")
    time.sleep(3)  # small delay before starting

    while image_count < images_per_gesture:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ROI
        x1, y1, x2, y2 = 320, 100, 620, 400
        roi = frame[y1:y2, x1:x2]

        # Preprocess
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Draw rectangle and label
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(display_frame, f"{gesture_name} ({image_count+1}/{images_per_gesture})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Gestures", display_frame)

        # Save image
        cv2.imwrite(os.path.join(gesture_path, f"{image_count+1}.jpg"), roi_resized)
        image_count += 1
        time.sleep(delay_between_images)  # small delay between captures

        # Exit if Enter key is pressed
        if cv2.waitKey(1) == 13:
            print("Interrupted by user")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("\n✅ Finished capturing all gestures")
cap.release()
cv2.destroyAllWindows()"""