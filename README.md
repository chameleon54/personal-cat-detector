# personal-cat-detector

Upload a photo and the app tells you the cat's breed, with a percentage for the top 3 matches.
Dogs get "It's a dog", people get "It's a human", and anything else gets "Not a cat".

- `Main_code/`: Python backend (FastAPI + TensorFlow). Details in [Main_code/README.md](Main_code/README.md).
- `UI/`: the website (plain HTML / CSS / JS). The backend serves it, so there is nothing to build.

## Run the website

Open a terminal (VS Code or CMD) in the project folder, then:

```
venv\Scripts\activate
pip install -r requirements.txt
cd Main_code
python -m uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/** in your browser.

- Run `pip install -r requirements.txt` again after pulling new changes: it now needs `opencv-python-headless`.
- The first start downloads the ImageNet weights (~14 MB), so it needs internet once.
- Loading the models takes a few seconds before the server says it is ready.

## Test bench (no website)

```
python Main_code\predict.py
```

Runs every image in `Main_code/Images/` through the same pipeline as the website and prints the JSON
for each one. It works from any folder.
