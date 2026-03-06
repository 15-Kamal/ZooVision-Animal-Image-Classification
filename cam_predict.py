import cv2
import numpy as np
import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = 'animal_classifier.keras'
DATA_DIR = 'data'
IMG_SIZE = (224, 224)
# Increase this threshold to reduce false positives (e.g., 0.80 for 80%)
CONFIDENCE_THRESHOLD = 85.0

# 1. Load Class Names
if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
    print(f"Loaded {len(class_names)} classes: {class_names}")
else:
    print("Error: Could not find 'data' folder.")
    exit()

# 2. Load Model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded!")

# 3. Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 4. Preprocess
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # 5. Predict
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0] # Use raw predictions directly
    
    confidence = 100 * np.max(score)
    predicted_class = class_names[np.argmax(score)]

    # 6. Display Logic with Threshold
    # If confidence is below the threshold, treat as "Unknown"
    if confidence > CONFIDENCE_THRESHOLD:
        label_text = f"{predicted_class}: {confidence:.1f}%"
        color = (0, 255, 0) # Green for high confidence
    else:
        label_text = "Unknown / No Animal Detected"
        color = (0, 0, 255) # Red for low confidence/unknown

    # Draw box and text
    cv2.rectangle(frame, (10, 10), (550, 60), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Animal Scanner', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()