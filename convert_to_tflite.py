import tensorflow as tf

print("1. Loading the massive Keras model...")
# Load your existing trained model
model = tf.keras.models.load_model('animal_classifier.keras')

print("2. Initializing TFLite Converter...")
# Convert the model to the lightweight TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional but highly recommended: Quantization
# This squishes the model's math from 32-bit floats down to 8-bit integers or 16-bit floats, 
# making the file size much smaller with almost zero loss in accuracy!
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("3. Compressing and converting...")
tflite_model = converter.convert()

print("4. Saving the new .tflite file...")
with open('animal_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print(" Success! Your model is now a lightweight TFLite file.")