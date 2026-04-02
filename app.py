import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ZooVision AI", page_icon="🦁", layout="centered")
IMG_SIZE = (224, 224)
DATA_DIR = 'data'
CONFIDENCE_THRESHOLD = 85.0  # From your OpenCV script!

# Try to load class names
try:
    class_names = sorted(os.listdir(DATA_DIR))
except FileNotFoundError:
    st.error(f"Could not find the '{DATA_DIR}' folder.")
    class_names = []

# --- 2. LOAD THE AI MODEL ---
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('animal_classifier.keras')

model = load_ai_model()

# --- 3. HELPER FUNCTION: PREDICT & DISPLAY ---
# We put this in a function so we can reuse it for BOTH uploads and webcam photos
def predict_and_display(image):
    with st.spinner("Analyzing pixels via TensorFlow..."):
        # Preprocess
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]
        
        confidence = float(np.max(score))
        predicted_class = class_names[np.argmax(score)]
        
        st.divider()
        
        # --- THRESHOLD LOGIC (From your OpenCV script) ---
        if (confidence * 100) > CONFIDENCE_THRESHOLD:
            st.success(f" **Animal Detected:** {predicted_class} ({confidence * 100:.1f}%)")
            
            st.subheader("Top Predictions")
            top_3_indices = np.argsort(score)[-3:][::-1]
            for i in top_3_indices:
                class_name = class_names[i]
                conf = float(score[i])
                st.write(f"**{class_name}**")
                st.progress(conf)
        else:
            st.error(" **Unknown / No Animal Detected**")
            st.write(f"The AI's highest guess was *{predicted_class}* at only {confidence * 100:.1f}%.")
            st.info(f"Our strict confidence threshold is set to {CONFIDENCE_THRESHOLD}%.")


# --- 4. THE WEB INTERFACE ---
st.title(" ZooVision: Deep Learning Classifier")
st.write("Choose how you want to analyze an image below!")

# Create two tabs for the two different input methods
tab1, tab2 = st.tabs(["📂 Upload Image", "📸 Webcam Scanner"])

# --- TAB 1: DRAG AND DROP ---
with tab1:
    st.subheader("Upload a Photo")
    uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Ready for analysis", use_container_width=True)
            if st.button(" Analyze Uploaded Image", type="primary", use_container_width=True):
                predict_and_display(image)


# --- TAB 2: WEBCAM SCANNER ---
with tab2:
    st.subheader("Live Webcam Scanner")
    st.write("Hold an animal picture up to your camera, or show your pet!")
    
    # Streamlit's built-in camera widget
    camera_photo = st.camera_input("Take a picture to analyze")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert('RGB')
        # The camera widget already displays the photo, so we just run the prediction
        predict_and_display(image)