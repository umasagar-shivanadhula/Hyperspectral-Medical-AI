# Unified HSI Medical Framework

**Unified Spectral–Spatial Multi-Classifier Framework for Hyperspectral Medical Image Analysis**

A complete end-to-end platform for two clinical tasks:
- **Task 1 — Tissue Perfusion Abnormality Detection** (SPECTRALPACA dataset)
- **Task 2 — Glioblastoma Brain Tumor Detection** (HistologyHSI-GB dataset)

---

## System Architecture

```
User → Frontend Dashboard → FastAPI Backend → ML Pipeline
                                               ├── Random Forest (spectral branch)
                                               ├── SVM (spatial branch)
                                               └── Softmax Fusion → Prediction
```

---

## Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the FastAPI Server

From the **project root** (Unified_HSI_Medical_Framework):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Server runs at http://localhost:8000
```

Or from the `backend` directory:

```bash
cd backend
python main.py
```

### 3. Open the Frontend

Open `frontend/index.html` in any modern browser.  
No build step required — pure HTML/CSS/JS.

**Demo Mode**: Click **"Run Demo with Synthetic Data"** on any analysis page  
to see the full pipeline visualization without uploading real data.

---

## API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| POST   | `/predict/perfusion`  | Perfusion state classification     |
| POST   | `/predict/tumor`      | Glioblastoma tumor detection       |
| GET    | `/`                   | API info                           |
| GET    | `/health`             | Health check                       |

### Example Request

```bash
curl -X POST http://localhost:8000/predict/perfusion \
  -F "file=@your_data.npy"
```

### Example Response

```json
{
  "task": "perfusion",
  "prediction": "Reduced Perfusion",
  "confidence": 87.3,
  "fusion_probabilities": {
    "Normal Perfusion": 8.2,
    "Reduced Perfusion": 87.3,
    "Abnormal Perfusion": 4.5
  },
  "classifiers": {
    "random_forest": { "prediction": "Reduced Perfusion", "confidence": 85.1 },
    "svm": { "prediction": "Reduced Perfusion", "confidence": 89.4 }
  },
  "spectral_signature": { "bands": [...], "tissue": [...], "healthy": [...] },
  "heatmap": [[...]],
  "ai_summary": "Decreased blood flow detected..."
}
```

---

## Training Models

```bash
# Train perfusion model
python ml_pipeline/training/train_perfusion.py

# Train tumor model
python ml_pipeline/training/train_tumor.py
```

Models are saved to `outputs/trained_models/`.

---

## Dataset Setup

### SPECTRALPACA (Perfusion)
Place data at:
```
datasets/perfusion_dataset/SPECTRALPACA/
  subject_01/
    normal_phase/  ← .npy cubes (H, W, 16)
    clamping_phase/
    recovery_phase/
  subject_02/
  ...
```

### HistologyHSI-GB (Tumor)
Place data at:
```
datasets/tumor_dataset/HistologyHSI-GB/
  P1/
    ROI_01_C01_T/
      raw
      raw.hdr
      darkReference
      whiteReference
  P2/
  ...
```

If datasets are not present, training scripts auto-generate synthetic data.

---

## Project Structure

```
Unified_HSI_Medical_Framework/
├── frontend/
│   ├── index.html          ← Home page
│   ├── perfusion.html      ← Perfusion analysis dashboard
│   ├── tumor.html          ← Tumor detection dashboard
│   ├── css/styles.css      ← Complete stylesheet
│   └── js/
│       ├── main.js         ← Home page animations
│       ├── api.js          ← API + demo data generators
│       ├── perfusion.js    ← Perfusion page logic
│       └── tumor.js        ← Tumor page logic
├── backend/
│   ├── main.py             ← FastAPI app
│   └── app/
│       ├── routes/predict.py   ← API endpoints
│       └── services/ml_service.py ← Inference pipeline
├── ml_pipeline/
│   ├── preprocessing/radiometric.py   ← Radiometric correction
│   ├── feature_extraction/
│   │   ├── spectral_features.py       ← Spectral statistics
│   │   └── spatial_features.py        ← GLCM texture features
│   ├── training/
│   │   ├── train_perfusion.py         ← Perfusion training
│   │   └── train_tumor.py             ← Tumor training
│   ├── fusion/softmax_fusion.py       ← Probability fusion
│   └── visualization/plots.py         ← Matplotlib charts
├── datasets/
├── outputs/trained_models/
├── configs/config.yaml
└── README.md
```

---

## ML Pipeline

```
Hyperspectral Cube (H×W×16)
  ↓
Radiometric Correction (dark/white reference)
  ↓
Patient-wise Train/Test Split (prevents data leakage)
  ↓
Patch Extraction (32×32×16)
  ↓
Feature Extraction
  ├── Spectral: mean, var, std, skewness, kurtosis, NDI   → (5B + B-1) dims
  └── Spatial: GLCM (contrast, energy, entropy, ...)       → (4 × bands × features) dims
  ↓
PCA Dimensionality Reduction (mitigates Hughes Phenomenon)
  ↓
Parallel Classifiers
  ├── Random Forest → spectral probability vector
  └── SVM (RBF)    → spatial probability vector
  ↓
Softmax Probability Fusion
  ↓
Final Prediction + Confidence + Heatmap
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | HTML5, CSS3, JavaScript, Chart.js, Plotly.js |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| ML | scikit-learn, NumPy, SciPy, Spectral Python |
| Visualization | Matplotlib, Plotly |
