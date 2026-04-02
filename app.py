import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ZooVision AI", page_icon="🦁", layout="centered")
CONFIDENCE_THRESHOLD = 85.0

#  UPDATE THIS LIST! It must have the EXACT SAME NUMBER of animals your model was trained on.
# If your model was trained on 10 animals, you must list all 10 here in alphabetical order!
# IMPORTANT: Update this list to match the exact animals your model was trained on!
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]

# --- 2. LOAD TFLITE MODEL ---
@st.cache_resource
def load_tflite_interpreter():
    # Load the lightweight TFLite model into memory
    interpreter = tf.lite.Interpreter(model_path="animal_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_interpreter()

# --- 3. INFERENCE ENGINE ---
def execute_tflite_inference(image, interpreter, class_names, threshold):
    with st.spinner("Executing fast TFLite inference..."):
        
        # A. Preprocess the Image (Resize & RGB conversion to prevent Alpha channel crashes)
        img_resized = image.convert('RGB').resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) 
        
        # TFLite strictly requires float32 data types
        input_data = np.array(img_array, dtype=np.float32)

        # B. Get Tensor Memory Addresses
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # C. Inject Data and Run Math
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # D. Extract the Results
        predictions = interpreter.get_tensor(output_details[0]['index'])
        probability_distribution = predictions[0]

        # E. The Gating Logic & Bug Fix
        max_confidence = float(np.max(probability_distribution)) * 100
        predicted_index = np.argmax(probability_distribution)
        
        # --- THE INDEX ERROR FIX ---
        if predicted_index >= len(class_names):
            st.error(f" Configuration Error: The AI predicted class index {predicted_index}, but your CLASS_NAMES list only has {len(class_names)} items.")
            st.warning("Please open `app.py`, go to line 11, and ensure you have listed ALL the animals your AI was trained on!")
            return # Stop the function from crashing
            
        predicted_label = class_names[predicted_index]

        if max_confidence >= threshold:
            st.success(f" Species Confirmed: {predicted_label}")
            st.metric(label="Mathematical Certainty", value=f"{max_confidence:.2f}%")
        else:
            st.error(" Unknown / Out-of-Distribution Data Detected")
            st.caption(f"Highest calculated probability was {max_confidence:.1f}%, which fails the {threshold}% security threshold.")

# --- 4. THE WEB UI (DUAL TABS) ---
st.title(" ZooVision AI")
st.write("Test the lightweight neural network using a saved photo or your live webcam.")
st.divider()

# Create the two tabs exactly as outlined in the technical report
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Webcam Scanner"])

# TAB 1: STATIC FILE UPLOAD
with tab1:
    st.subheader("Photo Analyzer")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Adding a unique key to the button prevents Streamlit from getting confused between the two tabs
        if st.button("Analyze Uploaded Image", type="primary", key="upload_btn"):
            execute_tflite_inference(image, interpreter, CLASS_NAMES, CONFIDENCE_THRESHOLD)

# TAB 2: WEBRTC CAMERA
with tab2:
    st.subheader("Live WebRTC Scanner")
    st.write("Ensure your browser has permission to access your camera.")
    
    # Streamlit handles the complex WebRTC connection automatically!
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        # The camera_input automatically renders the photo on screen, so we just pass it to the AI
        image = Image.open(camera_photo)
        
        if st.button("Analyze Camera Frame", type="primary", key="cam_btn"):
            execute_tflite_inference(image, interpreter, CLASS_NAMES, CONFIDENCE_THRESHOLD)