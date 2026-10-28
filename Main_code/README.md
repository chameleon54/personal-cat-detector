# Backend (Main_code)

Cat breed detection with Keras models trained on the Oxford-IIIT Pet dataset. See the
[root README](../README.md) for how to start the server and open the website.

## How an image is analyzed

1. **Is it a pet?** The stock ImageNet MobileNetV2 (`gatekeeper.py`). Cat + dog probability under 0.20 means no.
2. **Cat or dog?** Our 37-class Oxford pet model decides. (ImageNet alone calls a hairless Sphynx a dog.)
3. **Not a pet:** OpenCV's face detector decides `human`, otherwise `other`.
4. **Cat:** the breed model (`cat_breed_model_v2.keras`) returns the top 3 breeds. The result is averaged over
   the photo and its mirror image. If the top breed is under 40%, `uncertain` is `true`.

The logic lives in `classifier.py`, shared by the website (`main.py`) and `predict.py`.

## API

`POST /analyze` with a `multipart/form-data` field named `file` (JPG, PNG or WEBP, up to 25 MB).
The old `POST /predict` still works as an alias.

```json
{ "category": "cat",
  "predictions": [ {"breed": "Persian", "confidence": 0.88}, {"breed": "Maine Coon", "confidence": 0.08} ],
  "uncertain": false }
```
```json
{ "category": "dog", "message": "It's a dog" }
```

`category` is `cat`, `dog`, `human` or `other` (message "Not a cat"). `confidence` is 0 to 1.
Errors come back as `{"error": "..."}`: `400` for a file that is not a JPG/PNG/WEBP image, `413` for a file
over 25 MB, `500` for anything unexpected.

## Breed accuracy

Top-1 accuracy on the 1,183 cat images of the Oxford-IIIT Pet test split, choosing among the 12 cat breeds:

| Model | Top-1 | Top-3 |
|---|---|---|
| Original 37-class model | 80.6% | n/a |
| MobileNetV2, retrained cats only | 82.9% | 98.1% |
| **EfficientNetV2-B0, retrained cats only (current)** | **86.4%** | **98.6%** |

The weakest breeds are Ragdoll (66%, often mistaken for Birman), Russian Blue (73%) and British Shorthair (75%),
which look alike. That is why the website shows the top 3 and flags low confidence as "uncertain".


---

## Training a better breed model (optional)

`train_cats.py` trains a cat-only (12 breed) model with augmentation and
fine-tuning, then compares it to the original model on the test images.
Once `cat_breed_model_v2.keras` + `cat_class_names.txt` exist in this folder,
`main.py` uses them automatically (delete them to go back to the old model).

```
python train_cats.py                               # MobileNetV2 (~23 MB file)
python train_cats.py --backbone efficientnetv2b0   # stronger (~38 MB file), the current model
```

Training works on the CPU in the normal venv (no GPU needed). On a Ryzen 5 5600G, MobileNetV2 took
about 5 minutes and EfficientNetV2-B0 about 8 minutes. A GPU is optional - see below.
To keep the current model while experimenting, add `--out-dir some_folder`.

Options: `--epochs-head 10`, `--epochs-ft 20`, `--unfreeze 40`, `--mixed-precision`
(GPU only: faster, but the saved model is then slow on a CPU), `--out-dir <folder>`.

### Optional: training on the GPU (WSL2)
Skip this unless you want faster training or the EfficientNetV2 backbone.
TensorFlow 2.19 on native Windows only uses the CPU, so to use an NVIDIA GPU (e.g. RTX 3050) train inside WSL2.
The web server can keep running on Windows; only training needs WSL2.

1. Keep the NVIDIA driver up to date. In an **admin PowerShell**: `wsl --install -d Ubuntu`, then reboot.
2. In Ubuntu, `nvidia-smi` should list the GPU. Then:
   ```
   sudo apt update && sudo apt install -y python3-venv
   python3 -m venv ~/catenv && source ~/catenv/bin/activate
   pip install "tensorflow[and-cuda]==2.19.0" tensorflow-datasets==4.9.2 pillow numpy==2.1.3
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   The last line must list a GPU. Keep the same TensorFlow version as Windows so the saved model loads in both.
3. Train from the project folder (Windows drives appear under `/mnt/c`):
   ```
   cd "/mnt/c/Users/chair/OneDrive/Desktop/WORK/github rep/personal-cat-detector/Main_code"
   python train_cats.py --backbone efficientnetv2b0
   ```
   The first run downloads the Oxford-IIIT Pet dataset inside WSL (~0.8 GB).
4. Back on Windows, restart the server. It prints `Loaded breed model: cat_breed_model_v2.keras`.
