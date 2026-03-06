import matplotlib.pyplot as plt
import tensorflow as tf
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'      # Must match your folder name
IMG_SIZE = (224, 224)  # Size from your PDF
BATCH_SIZE = 32

print(f"Checking for data in: {os.path.abspath(DATA_DIR)}")

# 1. Load the dataset
# We only need the training set for visualization
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Success! Found classes: {class_names}")

    # 2. Visualize 9 Images
    print("Opening visualization window...")
    plt.figure(figsize=(10, 10))
    
    # Take 1 batch (32 images) and plot the first 9
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # Convert tensors to standard image format (uint8)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            
    # This command keeps the window open
    plt.show()

except Exception as e:
    print(f"Error loading data: {e}")
    print("Double check that your 'data' folder is in the same directory as this script!")