import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ZooVision AI", page_icon="🦁", layout="centered")
IMG_SIZE = (224, 224)
DATA_DIR = 'data'
CONFIDENCE_THRESHOLD = 85.0

# --- LOAD CLASS NAMES ---
if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
else:
    st.error(f"❌ '{DATA_DIR}' folder not found in repo")
    class_names = []

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists('animal_classifier.keras'):
        st.error("❌ Model file not found!")
        return None
    return tf.keras.models.load_model('animal_classifier.keras')

model = load_ai_model()

if model is None:
    st.stop()

# --- 3. PREDICTION FUNCTION ---
def predict_and_display(image):
    if not class_names:
        st.error("No class labels found.")
        return

    with st.spinner("Analyzing pixels via TensorFlow..."):
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]

        confidence = float(np.max(score))
        predicted_class = class_names[np.argmax(score)]

        st.divider()

        if (confidence * 100) > CONFIDENCE_THRESHOLD:
            st.success(f"Animal Detected: {predicted_class} ({confidence * 100:.1f}%)")

            st.subheader("Top Predictions")
            top_3_indices = np.argsort(score)[-3:][::-1]
            for i in top_3_indices:
                class_name = class_names[i]
                conf = float(score[i])
                st.write(f"**{class_name}**")
                st.progress(conf)
        else:
            st.error("Unknown / No Animal Detected")
            st.write(f"Highest guess: {predicted_class} ({confidence * 100:.1f}%)")
            st.info(f"Threshold: {CONFIDENCE_THRESHOLD}%")

# --- 4. UI ---
st.title("ZooVision: Deep Learning Classifier")
st.write("Choose how you want to analyze an image below!")

tab1, tab2 = st.tabs(["📂 Upload Image", "📸 Webcam Scanner"])

# --- TAB 1 ---
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        st.image(image, caption="Ready for analysis", use_container_width=True)

        if st.button("Analyze Uploaded Image"):
            predict_and_display(image)

# --- TAB 2 ---
with tab2:
    camera_photo = st.camera_input("Take a picture")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert('RGB')
        predict_and_display(image)