# Unified HSI Medical Framework v2.0

Spectral–Spatial AI Framework for Hyperspectral Medical Imaging  
**Glioblastoma Tumor Detection** · **Tissue Perfusion Analysis**

---

## Updated Project Structure

```
Unified_HSI_Medical_Framework/
│
├── config/
│   └── config.py               ← Central config (patch size, strides, RF/SVM params, paths)
│
├── logs/                       ← Auto-created rotating log files
│   └── hsi_framework.log
│
├── run.py                      ← Pipeline runner (serve / train / predict / health)
│
├── frontend/                   ← Static UI (unchanged — do not modify)
│   ├── index.html
│   ├── tumor.html
│   ├── perfusion.html
│   ├── css/styles.css
│   └── js/  api.js  main.js  tumor.js  perfusion.js
│
├── backend/
│   ├── main.py                 ← FastAPI app with structured logging
│   ├── requirements.txt
│   └── app/routes/predict.py   ← Upload validation + all 4 API endpoints
│
├── ml_pipeline/
│   ├── data_loader/
│   │   ├── hsi_loader.py       ← Safety-validated ENVI loader (§6 spec)
│   │   └── b2nd_loader.py
│   │
│   ├── preprocessing/
│   │   ├── patch_extraction.py ← Grid-based 3-D patches, band selection (§5 spec)
│   │   └── radiometric.py      ← Safe correction (only if both refs exist)
│   │
│   ├── feature_extraction/
│   │   ├── spectral_features.py
│   │   └── spatial_features.py
│   │
│   ├── fusion/
│   │   └── softmax_fusion.py
│   │
│   ├── prediction/
│   │   └── predict.py          ← Full pipeline: band sel → patches → features → RF+SVM → vote
│   │
│   ├── training/
│   │   ├── train_tumor.py      ← Config-driven, versioned models, eval JSON
│   │   └── train_perfusion.py
│   │
│   └── evaluation/
│       └── metrics.py
│
├── datasets/                   ← Place datasets here (NOT in repo)
│   ├── tumor_dataset/HistologyHSI-GB/
│   └── perfusion_dataset/SPECTRALPACA/
│
└── outputs/
    ├── trained_models/         ← *.pkl files (created by training)
    ├── evaluation_metrics/     ← *_metrics.json (created by training)
    └── prediction_results/     ← Per-prediction JSON logs (created at inference)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Train models
```bash
# Tumor (set dataset path first)
export HSI_TUMOR_DATASET=/path/to/HistologyHSI-GB
python run.py --train tumor

# Perfusion
export HSI_DATASETS_ROOT=/path/to/datasets
python run.py --train perfusion

# Both at once
python run.py --train both
```

### 3. Check readiness
```bash
python run.py --health
```

### 4. Start server
```bash
python run.py
# OR explicitly:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open frontend
Open `frontend/index.html` in a browser (or use Live Server in VS Code).

---

## Command-line Prediction
```bash
python run.py --predict tumor   path/to/file.hdr path/to/file.raw
python run.py --predict perfusion path/to/file.npy
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict/tumor` | Upload HSI file → tumor prediction |
| POST | `/predict/perfusion` | Upload HSI file → perfusion prediction |
| GET | `/predict/evaluation/tumor` | Training evaluation metrics |
| GET | `/predict/evaluation/perfusion` | Training evaluation metrics |
| GET | `/health` | Model readiness check |
| GET | `/docs` | Swagger UI |

**Allowed upload extensions:** `.hdr .raw .bin .b2nd .b2nb .bip .bil .bsq .npy .npz .h5`

---

## Configuration (config/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PATCH_SIZE` | 32 | Spatial patch size (H = W) |
| `STRIDE` | 16 | Grid stride for tumor task |
| `STRIDE_PERF` | 32 | Grid stride for perfusion task |
| `N_INFORMATIVE_BANDS` | 16 | Top-N bands by variance (0 = use all) |
| `USE_MAJORITY_VOTE` | True | Majority vote across patch predictions |
| `RF_TREES` | 300 | Random Forest estimators |
| `SVM_KERNEL` | rbf | SVM kernel |
| `N_PCA_TUMOR` | 25 | PCA components for tumor |
| `N_PCA_PERF` | 40 | PCA components for perfusion |
| `ENABLE_MODEL_VERSIONING` | True | Timestamped model backups |
| `LARGE_CUBE_ELEMENTS` | 2e9 | Warn if cube.size exceeds this |

---

## ML Pipeline Architecture

```
Hyperspectral Cube (H × W × B bands)
        │
        ▼  1. Radiometric Correction
        │     (dark/white refs if available; else per-band max normalisation)
        │
        ▼  2. Spectral Band Selection
        │     (top-N informative bands by spatial variance)
        │
        ▼  3. 32×32 Grid Patch Extraction
        │     (deterministic stride; patches are 3-D (32,32,B) — never flattened)
        │
        ▼  4. Per-patch Feature Extraction
        │     Spectral: mean, var, std, skew, kurt, NDI, slope, peak
        │     Spatial:  GLCM texture, gradient stats, LBP histogram
        │     Concatenated → 1-D feature vector per patch
        │
        ▼  5. StandardScaler  (fitted on training data)
        │
        ▼  6. PCA  (25 components for tumor / 40 for perfusion)
        │
        ▼  7. Random Forest predict_proba  →  RF probs  (N_patches × C)
           SVM            predict_proba  →  SVM probs (N_patches × C)
        │
        ▼  8. Per-patch Softmax Fusion
        │     fused_i = softmax( (RF_i + SVM_i) / 2 )
        │
        ▼  9. Majority Vote across patches
        │     Each patch votes for its argmax class.
        │     Class with most votes → Final label.
        │     (Average probabilities still computed for frontend charts)
        │
        ▼  Final Prediction + Confidence + Full JSON response
```

---

## Changes Applied (v2.0 — per Specification PDF)

### Section 5 — Pipeline Fixes
| # | Fix |
|---|-----|
| 5.1 | `patches.append(patch.flatten())` removed — patches stay `(P,P,B)` |
| 5.2 | All feature extraction receives 3-D `(H,W,B)` patches |
| 5.3 | Random sampling replaced with deterministic grid/stride extraction |
| 5.4 | Radiometric correction applied before patch extraction |
| 5.5 | Spectral band selection (top-N by variance) added before extraction |
| 5.6 | Majority voting across patch predictions replaces simple averaging |
| 5.7 | Training ≡ Inference: same preprocessing, extraction, scaler, PCA, fusion |
| 5.8 | `envi.open(hdr_path, raw_path)` enforced in loader |

### Section 6 — Safety Improvements
| # | Fix |
|---|-----|
| 6.1 | HSI loader validates: `.hdr` extension, no raw-as-hdr swap, stem match |
| 6.2 | Radiometric correction only applies when **both** references exist |
| 6.3 | Upload accepts only: `.hdr .raw .bin .b2nd .b2nb .bip .bil .bsq .npy .npz .h5` |
| 6.4 | Cube size warning logged when `cube.size > 2e9` |
| 6.5 | Model existence checked before `joblib.load()` → clear HTTP 503 if missing |
| 6.6 | Cube shape, patch count, model execution all logged via structured logger |

### Section 7 — Structural Improvements
| # | Fix |
|---|-----|
| 7.1 | `config/config.py` — central configuration for all parameters |
| 7.2 | `logs/` directory with rotating file handler |
| 7.3 | Model versioning with timestamps (`tumor_rf_20250306_143022.pkl`) |

### Section 9 — Engineering Improvements
| # | Fix |
|---|-----|
| 9.1 | Scaler + PCA saved alongside RF + SVM (was already done) |
| 9.2 | Feature dimension validation before inference |
| 9.3 | Exception handling for corrupt cubes (NaN/Inf check) |
| 9.6 | `/health` endpoint reports per-model file readiness |
| 9.7 | Structured logging replacing `print()` statements |
| 9.8 | Cube dimensions validated before processing |
| 9.9 | `run.py` — pipeline runner script |
| 9.10 | Prediction outputs saved to `outputs/prediction_results/` |
