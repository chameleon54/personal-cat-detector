# personal-cat-detector
i had a spare time so i make this random python code that can detect cat's breed by image using deep learning keras models and oxford iiit pet datasets
right now it doesnt have a GUI yet but if you want to run it just run predict.py.

Dont forget to install all from the requirements.txt using
"pip install -r requirements.txt"


---

## Training a better breed model (optional)

`train_cats.py` trains a cat-only (12 breed) model with augmentation and
fine-tuning, then compares it to the original model on the test images.
Once `cat_breed_model_v2.keras` + `cat_class_names.txt` exist in this folder,
`main.py` uses them automatically (delete them to go back to the old model).

```
python train_cats.py                               # MobileNetV2 (~10 MB), fine on a CPU
python train_cats.py --backbone efficientnetv2b0   # stronger (~25 MB), much slower on a CPU
```

Training on the CPU works out of the box in the normal venv (roughly 15-30 minutes
for MobileNetV2). A GPU is optional - see below.

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
