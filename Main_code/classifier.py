import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils import preprocess_pil, to_model_input, load_class_names, is_cat_breed, prettify_label
from gatekeeper import classify_category, pet_model, pet_class_names

# Shared by the API (main.py) and the CLI test bench (predict.py)

BASE_DIR = Path(__file__).resolve().parent
TOP_K = 3
UNCERTAIN_BELOW = 0.40   # top breed under this -> flag as mixed/uncertain

MESSAGES = {
    "dog": "It's a dog",
    "human": "It's a human",
    "other": "Not a cat",
}

# Load the breed model once.
# v2 = cat-only model from train_cats.py (does its own preprocessing).
# Fallback = original 37-class Oxford pet model (cats + dogs, expects 0..1 input).
V2_MODEL_PATH = BASE_DIR / "cat_breed_model_v2.keras"
if V2_MODEL_PATH.exists():
    model = tf.keras.models.load_model(V2_MODEL_PATH)
    class_names = load_class_names("cat_class_names.txt")
    prepare_input = to_model_input
    print(f"Loaded breed model: {V2_MODEL_PATH.name}", file=sys.stderr, flush=True)
else:
    model = pet_model   # already loaded by gatekeeper
    class_names = pet_class_names
    prepare_input = preprocess_pil
    print("Loaded breed model: cat_breed_model.h5 (fallback)", file=sys.stderr, flush=True)

cat_indices = [i for i, name in enumerate(class_names) if is_cat_breed(name)]

def predict_breeds(img):
    x = prepare_input(img)
    # Test-time augmentation: average the photo and its mirror image
    batch = np.concatenate([x, x[:, :, ::-1, :]], axis=0)
    probs = model.predict(batch, verbose=0).mean(axis=0)

    # Keep cat classes only and rescale so they add up to 1
    # (a no-op for the cat-only v2 model)
    cat_probs = probs[cat_indices]
    cat_probs = cat_probs / cat_probs.sum()

    order = np.argsort(cat_probs)[::-1][:TOP_K]
    return [
        {"breed": prettify_label(class_names[cat_indices[i]]),
         "confidence": round(float(cat_probs[i]), 4)}
        for i in order
    ]

def analyze(img):
    # Full pipeline for a PIL RGB image -> the /analyze JSON response
    category = classify_category(img)
    if category != "cat":
        return {"category": category, "message": MESSAGES[category]}

    predictions = predict_breeds(img)
    return {
        "category": "cat",
        "predictions": predictions,
        "uncertain": predictions[0]["confidence"] < UNCERTAIN_BELOW,
    }
