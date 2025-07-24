import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 224

def load_and_preprocess_image(image_path):
    """Load an image file and preprocess it for prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def load_class_names(filename="class_names.txt"):
    """Load class names from a .txt file."""
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]

def decode_prediction(pred, class_names):
    """Get top predicted class from the model's output."""
    top_index = np.argmax(pred)
    return class_names[top_index], pred[top_index]
