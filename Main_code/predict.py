import os
import tensorflow as tf
from utils import load_and_preprocess_image, load_class_names, decode_prediction


IMAGE_FOLDER = "test_images" #change this to your folder containing your cat images


model = tf.keras.models.load_model("cat_breed_model.h5")
class_names = load_class_names()


for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            image = load_and_preprocess_image(image_path)
            prediction = model.predict(image)[0]

            label, confidence = decode_prediction(prediction, class_names)
            print(f"{filename} → {label} ({confidence * 100:.2f}%)")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
