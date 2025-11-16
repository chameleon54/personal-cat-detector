# 🐱 Personal Cat Detector — Test Bench

This repository contains the randomness of I don't know what am I doing and backend server for the Personal Cat Detector project.  
Follow the steps below to install dependencies, activate the virtual environment, run the server, and test the prediction system.

---

## 📦 Installation

### 1. Install Dependencies

Before running anything, make sure to install all required packages:

```
pip install -r requirements.txt
```

To run the standalone prediction script:

```
python Main_code\predict.py
```

---

## 🌐 Running the Backend Server (FastAPI + Uvicorn)

### 1. Navigate to the `Main_code` Folder

Open VS Code terminal or Command Prompt:

```
cd Main_code
```

Example:

```
C:\..\github rep\personal-cat-detector\Main_code
```

### 2. Activate the Virtual Environment

```
..\venv\Scripts\activate
```

If successful, your terminal will show something like:

```
(venv) C:\your\path>
```

### 3. Start the FastAPI Server

```
python -m uvicorn main:app --reload
```

The server will be available at:

```
http://127.0.0.1:8000
```

---

## 🖥️ Running the Website / Frontend

If you want to run the project that includes its UI (HTML/CSS/JS):

1. Open the `UI` folder  
2. Use **Live Server** (VS Code extension)  
   **OR** serve the files using any static host  
3. Ensure the backend server is running  

The frontend communicates with the backend at:

```
http://127.0.0.1:8000/predict
```

---

## 🛠️ Notes

- Always activate the virtual environment before running any Python script.  
- If you add or update dependencies, reinstall them:

```
pip install -r requirements.txt
```

---

## ⭐ Contributing

Issues, suggestions, and pull requests are always welcome!
