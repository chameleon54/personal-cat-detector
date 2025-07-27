import os
from pathlib import Path
import tensorflow as tf
from utils import load_and_preprocess_image, load_class_names, decode_prediction

# Image folder and model path
IMAGE_FOLDER = Path("Main_code/Images")
MODEL_PATH = "Main_code/cat_breed_model.h5"

# Load model and class names
model = tf.keras.models.load_model(MODEL_PATH)
class_names = load_class_names()  # Defaults "Main_code/class_names.txt"

# image format
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# image paths based on supported extensions
image_paths = [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]

# Prepare images for batch prediction
images = []
valid_filenames = []

for image_path in image_paths:
    try:
        img = load_and_preprocess_image(str(image_path))  
        images.append(img)
        valid_filenames.append(image_path.name)
    except Exception as e:
        print(f"Error loading {image_path.name}: {e}")


if not images:
    print("No valid images found.")
    exit()

# Concatenate images into a single batch
batch = tf.concat(images, axis=0)

# Run batch prediction
predictions = model.predict(batch, verbose=0)

# Decode and print results
for filename, pred in zip(valid_filenames, predictions):
    label, confidence = decode_prediction(pred, class_names)
    print(f"{filename} → {label} ({confidence * 100:.2f}%)")
