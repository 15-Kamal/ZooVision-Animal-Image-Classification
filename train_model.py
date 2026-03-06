import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'      # Matches your folder name
IMG_SIZE = (224, 224)  # 224x224 as per requirements
BATCH_SIZE = 32
EPOCHS = 10            # Number of training rounds

print(f"TensorFlow Version: {tf.__version__}")
print("1. Loading Data...")

# Load Training Data (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load Validation Data (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# Optimize performance (Keep data in memory for speed)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n2. Building Model (Transfer Learning with MobileNetV2)...")

# Download MobileNetV2 (Pre-trained on ImageNet)
# include_top=False removes the original 1000-class output layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False, 
    weights='imagenet'
)

# Freeze the base model so we don't destroy pre-trained patterns
base_model.trainable = False 

# Add custom layers for your 15 animals
model = models.Sequential([
    # Rescale pixel values from [0, 255] to [-1, 1] for MobileNet
    layers.Rescaling(1./127.5, offset=-1, input_shape=IMG_SIZE + (3,)),
    
    # Data Augmentation: Randomly flip/rotate images to prevent overfitting
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # The Pre-trained Base
    base_model,
    
    # Flatten results into a 1D vector
    layers.GlobalAveragePooling2D(),
    
    # Dropout: Randomly turn off 20% of neurons to prevent memorization
    layers.Dropout(0.2),
    
    # Output Layer: 15 neurons for 15 classes
    layers.Dense(15, activation='softmax') 
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n3. Training Started... (This may take a few minutes)")
# This is the actual "Learning" process
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("\n4. Saving Model...")
model.save('animal_classifier.keras')
print("Model saved as 'animal_classifier.keras'!")

# --- PLOTTING RESULTS ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()