from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from utils import load_image_from_bytes
from classifier import analyze

app = FastAPI()

# Allow all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR.parent / "UI"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Plain def (not async) so FastAPI runs the model in a worker thread
@app.post("/analyze")
@app.post("/predict")  # old name, kept so existing callers still work
def analyze_image(file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(content={"error": "Image is too large (max 25 MB)."}, status_code=413)

    try:
        img = load_image_from_bytes(data)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    try:
        return analyze(img)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Serve the web UI from the same server: http://127.0.0.1:8000/
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/index/index.html")

app.mount("/", StaticFiles(directory=UI_DIR), name="ui")
