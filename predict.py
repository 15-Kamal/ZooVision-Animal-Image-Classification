import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
IMG_SIZE = (224, 224)
class_names = sorted(os.listdir(DATA_DIR)) 

print("1. Loading your saved model...")
model = tf.keras.models.load_model('animal_classifier.keras')

print("2. Picking a random image for testing...")
random_class = random.choice(class_names)
random_folder = os.path.join(DATA_DIR, random_class)
random_image = random.choice(os.listdir(random_folder))
img_path = os.path.join(random_folder, random_image)

print(f"   Selected Image: {img_path}")

# 3. Preprocessing
img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

# 4. PREDICTION TIME!
predictions = model.predict(img_array)

# --- THE FIX IS HERE ---
# We take the raw predictions directly because the model already used Softmax
score = predictions[0] 

predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

# 5. Show the Result
print(f"\n--- RESULT ---")
print(f"ACTUAL Class:    {random_class}")
print(f"PREDICTED Class: {predicted_class}")
print(f"Confidence:      {confidence:.2f}%")

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Actual: {random_class}\nPredicted: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()