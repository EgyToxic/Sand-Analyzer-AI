# 🏗️ AI Sand Suitability Inspector

An intelligent Deep Learning system designed to analyze soil images and determine their suitability for construction purposes. Powered by **MobileNetV2**.

## 📊 Project Overview
In civil engineering, sand quality is critical for structural stability. This project automates the inspection process using Computer Vision to distinguish between:
* ✅ **Acceptable Sand:** (River sand, clean pit sand)
* ❌ **Unacceptable Soil:** (Clay, organic soil, peat, etc.)

## 🚀 Key Features
* **High Accuracy:** ~92% accuracy on test data.
* **Architecture:** MobileNetV2 (Transfer Learning).
* **Safety First:** Includes a confidence threshold system for engineering decisions.
* **Lightweight:** Optimized for deployment (~9MB model size).

## 📂 Structure
* `api/`: Contains the Flask web server for inference.
* `training/`: Contains the model architecture definition.
* `requirements.txt`: Project dependencies.

## 🛠️ How to Run (Locally)
1.  **Clone the repo:**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the API:**
    ```bash
    cd api
    python main.py
    ```

4.  **Test:**
    Send a POST request to `http://localhost:8080/predict` with an image file.

## 📈 Results
The model was trained on a dataset of ~6,000 images using a Two-Phase training strategy (Head Training + Fine-Tuning).

| Metric | Score |
|OSS|OSS|
| **Accuracy** | 92% |
| **Precision (Acceptable)** | 97% |
| **Recall (Unacceptable)** | 90% |

---
*Developed as a Graduation Project for Civil Engineering AI Integration.*