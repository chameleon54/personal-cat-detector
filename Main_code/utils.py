import io
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 224

# The 12 cat breeds in Oxford-IIIT Pet (the other 25 classes are dogs)
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
    "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
}

def is_cat_breed(name):
    return name in CAT_BREEDS

def prettify_label(name):
    # "Maine_Coon" -> "Maine Coon"
    return name.replace("_", " ")

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

def load_image_from_bytes(data):
    # Read an uploaded image in memory, fixing phone-photo rotation (EXIF).
    # Raises ValueError if it isn't a JPG/PNG/WEBP image.
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise ValueError("That file isn't a readable image.")
    if img.format not in ALLOWED_FORMATS:
        raise ValueError("Unsupported image type. Please upload a JPG, PNG or WEBP.")
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')

def preprocess_pil(img):
    # Same scaling as training (data_loader.preprocess): 224x224, 0..1
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def to_model_input(img):
    # For v2 models that do their own preprocessing: 224x224, raw 0..255
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return preprocess_pil(img)

def load_class_names(filename="class_names.txt"):
    base_dir = Path(__file__).resolve().parent
    class_file = base_dir / filename
    with open(class_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def decode_prediction(pred, class_names):
    top_index = np.argmax(pred)
    return class_names[top_index], pred[top_index]
