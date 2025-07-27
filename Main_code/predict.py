import os
from pathlib import Path
import tensorflow as tf
from utils import load_and_preprocess_image, load_class_names, decode_prediction

#image folder and model path
IMAGE_FOLDER = Path("Main_code/Images")
model = tf.keras.models.load_model("Main_code/cat_breed_model.h5")
class_names = load_class_names()

#image format files
image_paths = [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

#changing the process into a single batch instead of one by one 
images = []
valid_paths = []

for image_path in image_paths:
    try:
        img = load_and_preprocess_image(str(image_path))
        images.append(img)
        valid_paths.append(image_path.name)
    except Exception as e:
        print(f"Error loading {image_path.name}: {e}")

if not images:
    print("No valid images found.")
    exit()

# Stack into a batch
batch = tf.concat(images, axis=0)  


predictions = model.predict(batch)


for filename, pred in zip(valid_paths, predictions):
    label, confidence = decode_prediction(pred, class_names)
    print(f"{filename} → {label} ({confidence * 100:.2f}%)")
