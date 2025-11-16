from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import shutil
import os
from utils import load_and_preprocess_image, load_class_names, decode_prediction
from pathlib import Path
import tempfile

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Allow all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and classes once
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cat_breed_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = load_class_names()  # e.g. from Main_code/class_names.txt

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Load and preprocess image
        img_tensor = load_and_preprocess_image(tmp_path)
        prediction = model.predict(img_tensor, verbose=0)[0]
        label, confidence = decode_prediction(prediction, class_names)

        return JSONResponse(content={
            "filename": file.filename,
            "label": label,
            "confidence": f"{confidence * 100:.2f}%"  # as percent
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(tmp_path)
