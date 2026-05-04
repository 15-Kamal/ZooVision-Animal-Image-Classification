import os
import random
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="ZooVision AI", page_icon="🦁", layout="wide")

CONFIDENCE_THRESHOLD = 95.0
MARGIN_THRESHOLD = 40.0

# IMPORTANT: Ensure this folder name matches exactly where your training images are saved!
DATASET_DIR = "data"

CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]

# --- 2. LOAD BOTH AI BRAINS ---
@st.cache_resource
def load_custom_model():
    interpreter = tf.lite.Interpreter(model_path="animal_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_global_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

custom_interpreter = load_custom_model()
global_model = load_global_model()

# --- 3. DATASET EVIDENCE PULLER ---
def get_dataset_example(predicted_class):
    """Pulls a random image from the training dataset for visual proof."""
    class_folder = os.path.join(DATASET_DIR, predicted_class)
    
    if os.path.exists(class_folder) and os.path.isdir(class_folder):
        # Find all valid images in that folder
        valid_extensions = ('.jpg', '.jpeg', '.png')
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(valid_extensions)]
        
        if images:
            random_image = random.choice(images)
            return os.path.join(class_folder, random_image)
    
    return None # Return nothing if folder or images are missing

# --- 4. INFERENCE ENGINE ---
def process_image(image):
    img_resized = image.convert('RGB').resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    return preprocess_input(img_array)

def execute_dual_inference(image):
    input_data = process_image(image)
    
    # -----------------------------------------
    # BRAIN 1: Run Custom TFLite Model
    # -----------------------------------------
    input_details = custom_interpreter.get_input_details()
    output_details = custom_interpreter.get_output_details()
    
    custom_interpreter.set_tensor(input_details[0]['index'], input_data)
    custom_interpreter.invoke()
    
    predictions = custom_interpreter.get_tensor(output_details[0]['index'])[0]
    
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3_labels = [CLASS_NAMES[i] for i in top_3_indices]
    top_3_probs = [float(predictions[i]) * 100 for i in top_3_indices]
    
    custom_confidence = top_3_probs[0]
    runner_up_confidence = top_3_probs[1]
    custom_guess = top_3_labels[0]
    confidence_margin = custom_confidence - runner_up_confidence
    
    custom_df = pd.DataFrame({"Probability (%)": top_3_probs}, index=top_3_labels)
    
    # -----------------------------------------
    # BRAIN 2: Run Global Fallback Model
    # -----------------------------------------
    global_predictions = global_model.predict(input_data, verbose=0)
    decoded_global = decode_predictions(global_predictions, top=3)[0]
    
    global_labels = [item[1].replace('_', ' ').title() for item in decoded_global]
    global_probs = [float(item[2]) * 100 for item in decoded_global]
    global_guess = global_labels[0]
    
    global_df = pd.DataFrame({"Probability (%)": global_probs}, index=global_labels)
    
    # -----------------------------------------
    # DASHBOARD UI
    # -----------------------------------------
    col1, col2 = st.columns([1, 1.3])
    
    with col1:
        st.image(image, caption="Your Uploaded Image", use_container_width=True)
        
        # --- NEW: SHOW VISUAL EVIDENCE FROM DATASET ---
        is_confident = custom_confidence >= CONFIDENCE_THRESHOLD
        is_decisive = confidence_margin >= MARGIN_THRESHOLD
        
        if is_confident and is_decisive:
            reference_image_path = get_dataset_example(custom_guess)
            if reference_image_path:
                st.divider()
                st.markdown("#### 🔍 Visual Evidence")
                st.caption(f"The neural network matched your image to this sample from the **{custom_guess}** training data:")
                
                # Load and display the dataset image
                ref_img = Image.open(reference_image_path)
                st.image(ref_img, caption=f"Dataset Source: /{DATASET_DIR}/{custom_guess}/", use_container_width=True)
        
    with col2:
        st.markdown("### 🧠 AI Analysis Engine")
        
        if is_confident and is_decisive:
            st.success(f"### ✅ Primary Database Match: **{custom_guess}**")
            st.caption(f"Network Confidence: {custom_confidence:.2f}% (Margin: {confidence_margin:.1f}%)")
        else:
            st.warning("⚠️ **Subject not in primary 15-animal database (Out-of-Distribution).**")
            st.info(f"### 🌍 Global AI Fallback Match: **{global_guess}**")
            if not is_confident:
                st.caption(f"Reason: Custom AI highest confidence was only {custom_confidence:.1f}% (Required: {CONFIDENCE_THRESHOLD}%).")
            elif not is_decisive:
                st.caption(f"Reason: Custom AI was indecisive. Margin was only {confidence_margin:.1f}% (Required: {MARGIN_THRESHOLD}%).")
        
        st.divider()
        st.markdown("### 📊 Probability Data Visualization")
        
        st.markdown("#### 1. Custom Network (15-Class Limit)")
        table_col, chart_col = st.columns(2)
        with table_col:
            st.table(custom_df.style.format("{:.2f}%")) 
        with chart_col:
            st.bar_chart(custom_df)
            
        if not (is_confident and is_decisive):
            st.divider()
            st.markdown("#### 2. Global Network (1,000-Class Limit)")
            table_col2, chart_col2 = st.columns(2)
            with table_col2:
                st.table(global_df.style.format("{:.2f}%"))
            with chart_col2:
                st.bar_chart(global_df)

# --- 5. THE WEB APP ---
st.title("🦁 ZooVision AI: Dual-Brain Architecture")
st.write("Upload an image or use your camera. The AI will check the primary 15-animal database. If it doesn't recognize it, the Global Fallback AI will step in!")
st.divider()

# Re-introducing the Tabs!
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Webcam Scanner"])

# TAB 1: FILE UPLOAD
with tab1:
    uploaded_file = st.file_uploader("Upload an animal image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with st.spinner("Initializing Dual-Brain Analysis..."):
            execute_dual_inference(image)

# TAB 2: LIVE WEBCAM
with tab2:
    st.info("Ensure your browser has permission to access your camera.")
    
    # We create 3 invisible columns: 20% empty space, 60% camera space, 20% empty space
    spacer_left, cam_col, spacer_right = st.columns([0.2, 0.6, 0.2])
    
    # We place the camera ONLY inside the middle 60% column
    with cam_col:
        camera_photo = st.camera_input("Take a picture of an animal")
    
    # The inference logic stays outside the columns so the results still use the full screen
    if camera_photo is not None:
        image = Image.open(camera_photo)
        with st.spinner("Initializing Dual-Brain Analysis..."):
            execute_dual_inference(image)