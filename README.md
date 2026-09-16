# Handwritten Character Recognition using Deep Learning (CNN)

A beginner-friendly but fully functional Python mini project that recognizes handwritten **digits and English characters** from images.

For formal submission, see the full report: `TECHNICAL_REPORT.md`

This project is suitable for:
- College mini project
- Resume project
- GitHub portfolio
- Basic AI/ML demo

---

## 1. Project Objective

Build an end-to-end system that:
- Trains a CNN model on handwritten characters
- Predicts character/digit from uploaded image
- Shows confidence score
- Saves and reloads trained model
- Displays training accuracy/loss graphs
- Supports OCR mode for image/PDF text extraction (no trained CNN required)

---

## 2. Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Flask (web UI)
- RapidOCR (word-level OCR fallback)
- PyMuPDF (PDF to image rendering for OCR)
- wordfreq (OCR post-correction for likely misspellings)

---

## 3. Project Folder Structure

```text
Handwritten_Character_Recognition/
│── dataset/
│── model/
│   │── __init__.py
│   │── cnn_model.py
│   │── data_loader.py
│   │── preprocessing.py
│── static/
│   │── uploads/
│   │── style.css
│── templates/
│   │── index.html
│── saved_model/
│── app.py
│── train.py
│── predict.py
│── requirements.txt
│── README.md
```

`dataset/` is intentionally empty at first. You can add your dataset manually later.

---

## 4. Installation

### Step 1: Open terminal in project folder

```bash
cd Handwritten_Character_Recognition
```

### Step 2: Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

- Windows (PowerShell):
```bash
venv\Scripts\Activate.ps1
```

- Linux/Mac:
```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Dataset Setup (Manual)

This project supports **MNIST**, **EMNIST**, and **Custom dataset**.

## A) MNIST (digits 0-9)

No manual files required. TensorFlow downloads MNIST automatically.

Train command:

```bash
python train.py --dataset_type mnist
```

## B) EMNIST (digits + alphabets)

Place EMNIST CSV files inside:

```text
dataset/emnist/
```

Expected file names (example):

- `emnist-balanced-train.csv`
- `emnist-balanced-test.csv`
- optional mapping file: `emnist-balanced-mapping.txt`

Train command:

```bash
python train.py --dataset_type emnist --dataset_dir dataset
```

## C) Custom Handwritten Dataset

Create folders by class name:

```text
dataset/custom/
│── 0/
│── 1/
│── A/
│── B/
```

Put images in each class folder. Supported formats:
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

Train command:

```bash
python train.py --dataset_type custom --dataset_dir dataset
```

---

## 6. Train the CNN Model

Basic training:

```bash
python train.py --dataset_type mnist
```

Custom options example:

```bash
python train.py --dataset_type custom --epochs 15 --batch_size 64 --learning_rate 0.001
```

After training, these files are created:
- `saved_model/handwritten_cnn.keras` (trained model)
- `saved_model/class_names.json` (label names)
- `saved_model/training_summary.json` (final metrics)
- `static/training_curves.png` (accuracy/loss graph)

---

## 7. Predict from Command Line

```bash
python predict.py path_to_image.png
```

Example output:
- Predicted Character
- Confidence Score
- Raw class probabilities

---

## 8. Run Web UI (Flask)

Start app:

```bash
python app.py
```

The server binds to `0.0.0.0` by default, uses port `5000` locally, and honors a deployment-provided `PORT` value. For example, in PowerShell:

```powershell
$env:PORT=5050
python app.py
```

Or on Windows (double-click friendly):

```bash
run_web_preview.bat
```

Open browser:

```text
http://localhost:5000
```

UI Features:
- Upload handwritten image
- Click **Predict**
- See predicted character and confidence
- View latest training curve image (if available)
- Switch between:
  - **Character Mode (CNN)** for single character prediction
  - **OCR Mode** for whole-word/line extraction from images and PDFs

---

## 9. How Handwritten Character Recognition Works

1. Input image is provided (uploaded or dataset sample).
2. Image is converted to grayscale.
3. Image is resized (default `28x28`) and normalized to `[0, 1]`.
4. CNN extracts visual patterns (edges, strokes, shapes).
5. Final softmax layer outputs class probabilities.
6. Class with highest probability becomes prediction.

---

## 10. CNN Workflow in This Project

The model uses:
- `Conv2D`: learns local patterns (strokes/curves)
- `MaxPooling2D`: reduces image size while keeping important features
- `Flatten`: converts feature maps into 1D vector
- `Dense`: learns class-level decision boundaries
- `Dropout`: reduces overfitting
- `Softmax`: gives probability for each class

Loss function:
- `sparse_categorical_crossentropy`

Optimizer:
- `Adam`

---

## 11. Image Preprocessing Steps

For prediction and custom data:
- Read image using OpenCV
- Convert to grayscale
- Auto-invert when image has bright background (paper style)
- Resize to model input size
- Gaussian blur for mild denoising
- Normalize pixel values (`0` to `1`)
- Add channel dimension for CNN (`H x W x 1`)

---

## 12. Model Training Process

1. Load selected dataset (`mnist`, `emnist`, `custom`)
2. Prepare train and test sets
3. Build CNN model architecture
4. Train model for given epochs
5. Evaluate on test data
6. Save model and class names
7. Plot and save accuracy/loss curves

---

## 13. Real-World Applications

- Bank cheque digit recognition
- Postal code and address reading
- Form digitization in offices
- Student answer-sheet processing
- License plate and document OCR preprocessing
- Historical manuscript digitization

---

## 14. Future Improvements

- Data augmentation for better generalization
- Use Streamlit as alternative UI
- Add webcam input support
- Add confusion matrix and classification report
- Deploy on cloud (Render/Heroku/AWS)
- Add model quantization for mobile devices

---

## 15. Important Notes

- Dataset paths are **not hardcoded**; use command-line arguments.
- `dataset/` can remain empty until you add data manually.
- If model files are missing, web app shows helpful message.
- For best results, keep training and prediction image style similar.

---

## 16. Quick Start Commands

```bash
# 1) Install packages
pip install -r requirements.txt

# 2) Train (MNIST)
python train.py --dataset_type mnist

# 3) Start web app
python app.py
```

To use OCR mode from the UI, upload an image or PDF and select **OCR Mode (Words / PDF)**.
