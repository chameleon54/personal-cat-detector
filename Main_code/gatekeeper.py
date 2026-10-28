import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils import preprocess_pil, load_class_names, is_cat_breed

# First check before breed classification: is this a cat, dog, human or something else?
#   1. Stock ImageNet MobileNetV2 decides "is this a cat/dog at all?"
#      (weights download ~14MB on first run)
#   2. Our Oxford pet model (37 cat + dog breeds) decides cat vs dog.
#      ImageNet alone gets hairless cats wrong: it has no Sphynx class and
#      calls them hairless dogs.
#   3. Not an animal -> OpenCV's bundled face detector decides human vs other.

BASE_DIR = Path(__file__).resolve().parent
IMAGENET_SIZE = 224

# ImageNet class indices
CAT_CLASSES = [281, 282, 283, 284, 285, 287]   # tabby, tiger cat, Persian, Siamese, Egyptian cat, lynx
DOG_CLASSES = list(range(151, 269))            # Chihuahua ... Mexican hairless

# Tune these if real photos get misclassified
ANIMAL_THRESHOLD = 0.20      # min ImageNet cat+dog probability to count as a pet
CAT_SHARE_THRESHOLD = 0.50   # min pet-model cat probability to call it a cat
FACE_MIN_SIZE_RATIO = 0.08   # ignore "faces" smaller than this fraction of the image

_imagenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")
_face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 37-class Oxford pet model (also the fallback breed model in main.py)
pet_model = tf.keras.models.load_model(BASE_DIR / "cat_breed_model.h5")
pet_class_names = load_class_names()
_pet_cat_indices = [i for i, name in enumerate(pet_class_names) if is_cat_breed(name)]

def animal_score(img):
    # Summed ImageNet probability for cat and dog classes
    x = np.array(img.resize((IMAGENET_SIZE, IMAGENET_SIZE)), dtype=np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x[np.newaxis])
    probs = _imagenet_model.predict(x, verbose=0)[0]
    return float(probs[CAT_CLASSES].sum() + probs[DOG_CLASSES].sum())

def cat_share(img):
    # Pet-model probability that this pet is a cat (vs a dog)
    probs = pet_model.predict(preprocess_pil(img), verbose=0)[0]
    return float(probs[_pet_cat_indices].sum())

def has_human_face(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    min_side = int(min(gray.shape) * FACE_MIN_SIZE_RATIO)
    faces = _face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
    )
    return len(faces) > 0

def classify_category(img):
    # Returns "cat", "dog", "human" or "other" for a PIL RGB image
    if animal_score(img) >= ANIMAL_THRESHOLD:
        return "cat" if cat_share(img) >= CAT_SHARE_THRESHOLD else "dog"
    if has_human_face(img):
        return "human"
    return "other"
