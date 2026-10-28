# Test bench: runs every image in Main_code/Images through the same
# pipeline as the web API and prints the /analyze JSON for each one.
#   python Main_code/predict.py      (works from any folder)
import json
from pathlib import Path
from utils import load_image_from_bytes
from classifier import analyze

IMAGE_FOLDER = Path(__file__).resolve().parent / "Images"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

image_paths = sorted(p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
if not image_paths:
    print("No valid images found.")
    raise SystemExit

results = []
for image_path in image_paths:
    try:
        img = load_image_from_bytes(image_path.read_bytes())
        results.append({"filename": image_path.name, **analyze(img)})
    except Exception as e:
        results.append({"filename": image_path.name, "error": str(e)})

print(json.dumps(results, indent=2))
