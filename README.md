# 🫁 CNN-Based Tuberculosis Classification System

> Detect tuberculosis from chest X-ray images using deep learning — with transfer learning, uncertainty estimation, explainability, and a one-command training pipeline.

<p align="center">
  <img src="Screenshot 2026-07-15 193837.png" alt="Project pipeline" width="90%">
</p>

![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/use-research%20%2F%20educational-important)

> ⚠️ **Research and educational use only. This is NOT a medical device and must not be used for diagnosis or any clinical decision. Chest X-ray interpretation requires a qualified radiologist.**

---

## 📖 Overview

This project classifies chest X-rays as **Normal** or **Tuberculosis** using a convolutional neural network. It started as a simple from-scratch CNN and was rebuilt into a full, tested pipeline: you point it at one folder of images and it automatically splits the data, trains a model, evaluates it, and serves predictions through a web app.

The app accepts both **file uploads** and **mobile camera** input, and for every prediction it shows not just a label but a **confidence interval** (how sure the model is) and a **Grad-CAM heatmap** (where the model looked).

## 🖼️ The data

Trained on the **Tuberculosis (TB) Chest X-ray Database** (Qatar University, University of Dhaka, and collaborators) — 3,500 TB and 3,500 Normal 512×512 chest X-rays.

<p align="center">
  <img src="assets/sample_xrays.png" alt="Sample chest X-rays" width="70%">
</p>

## ✨ Key features

| Feature | What it does |
|---|---|
| 🔀 **One-command pipeline** | Paste a single folder path → auto stratified train/val/test split → train → save |
| 🧠 **Transfer learning** | MobileNetV2 pretrained on ImageNet, not a fragile from-scratch CNN |
| 📊 **Uncertainty** | Monte-Carlo dropout gives a probability *and* a 95% confidence interval per prediction |
| 🔥 **Explainability** | Grad-CAM heatmaps reveal which lung regions drove the decision |
| 📷 **Camera input** | Upload a file or capture with your phone camera (Streamlit) |
| 📈 **Real evaluation** | Accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix, threshold sweep |
| ✅ **Tested** | 38 unit tests; the core logic runs in CI with no GPU or TensorFlow needed |

## 🚀 Quick start

```bash
# 1. clone and enter
git clone https://github.com/<your-username>/CNN-Based-Tuberculosis-Classification-System.git
cd CNN-Based-Tuberculosis-Classification-System

# 2. (recommended) create an isolated environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. install
pip install -r requirements.txt
```

### Train, evaluate, run — three commands

You only supply the path to your dataset folder. Everything else is automatic.

```bash
# ① split + train + save the model  (creates model/tb_model.keras)
python scripts/train.py --data "path/to/TB_Chest_Radiography_Database"

# ② evaluate on the held-out test set  (writes reports/)
python scripts/evaluate.py --split-dir data_split

# ③ launch the web app  (upload + camera)
streamlit run app.py
```

> 💡 Training on a CPU is slow (a few hours for 7,000 images). A GPU — including Google Colab's free tier — cuts this to minutes.

### Dataset layout

Point `--data` at a folder with **one subfolder per class** (exactly what Kaggle gives you):

```
TB_Chest_Radiography_Database/
├── Normal/          *.png
└── Tuberculosis/    *.png
```

The training script auto-creates a stratified, seeded 70/15/15 split under `data_split/` — no manual sorting.

## 📱 The app

<!-- 👉 After running `streamlit run app.py`, take screenshots and replace these lines: -->
<!-- ![Upload prediction](assets/screenshot_prediction.png) -->
<!-- ![Grad-CAM heatmap](assets/screenshot_gradcam.png) -->

> **Add your own screenshots here** once the app is running — one of a prediction with the confidence interval, and one of the Grad-CAM heatmap. Save them into `assets/` and un-comment the two lines above in the README source.

The app has two tabs:
- **📁 Upload** — drag in a chest X-ray (`.png/.jpg`); most reliable.
- **📷 Camera** — capture with your device camera (works on mobile browsers). Great for demos; note that camera glare/angle lower reliability, so it is labelled demo-only in the UI.

## 📊 Results

Run `python scripts/evaluate.py --split-dir data_split` and paste your numbers here:

| Metric | Score |
|---|---|
| Accuracy | 0.9778 |
| Precision | 1.0 |
| Recall (sensitivity) | 0.8667 |
| Specificity | 1.0 |
| F1 | 0.9286 |
| ROC-AUC | 0.9989 |

The script also writes `reports/confusion_matrix.png`, `reports/roc_curve.png`, and `reports/threshold_sweep.csv` — add the plots here to make this section shine.

## 🗂️ Project structure

```
tb-classifier/
├── tb_lib/                 core library (tested, mostly TensorFlow-free)
│   ├── config.py           image size, class names, paths
│   ├── dataset.py          auto-split one folder → train/val/test
│   ├── preprocessing.py    image validation + tensor conversion
│   ├── architecture.py     MobileNetV2 model definition
│   ├── model.py            model loader (only file importing TensorFlow)
│   ├── predict.py          MC-dropout prediction + uncertainty
│   ├── gradcam.py          Grad-CAM heatmaps (Keras 3 compatible)
│   └── metrics.py          confusion matrix, ROC-AUC, threshold sweep
├── scripts/
│   ├── train.py            ① split + train + save
│   └── evaluate.py         ② evaluate + write reports
├── app.py                  ③ Streamlit web app (upload + camera)
├── api.py                  FastAPI backend (optional)
├── flask_app.py            Flask app (optional)
├── tests/                  38 pytest cases
└── assets/                 images used in this README
```

## 🧠 How it works

1. **Preprocessing** resizes each X-ray to 224×224 and scales pixels to `[0,1]`, after validating the file is a real, safe image.
2. **MobileNetV2** (pretrained on ImageNet) extracts features; a small trained head outputs `P(Tuberculosis)`.
3. **Monte-Carlo dropout** keeps dropout active at inference and runs 30 stochastic passes, so each prediction has a mean probability and a spread → a confidence interval.
4. **Grad-CAM** back-propagates the score to the last convolutional layer to highlight the regions that mattered — a check that the model looks at lungs, not text or borders.

## 🛠️ Tech stack

Python · TensorFlow / Keras 3 · MobileNetV2 · Streamlit · FastAPI · Flask · NumPy · Pillow · Matplotlib · pytest

## 📚 Dataset citation

If you use the dataset, please cite:

> T. Rahman, A. Khandakar, M. A. Kadir, K. R. Islam, K. F. Islam, Z. B. Mahbub, M. A. Ayari, M. E. H. Chowdhury, *"Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization,"* IEEE Access, vol. 8, pp. 191586–191601, 2020. DOI: 10.1109/ACCESS.2020.3031384.

## ⚖️ License

Released under the MIT License — see [LICENSE](LICENSE). The dataset is subject to its own terms; see the citation above.

## 🙏 Disclaimer

This project is for learning and demonstration only. It has **no clinical validation** and must never be used to make real medical decisions. Always consult a qualified healthcare professional.
