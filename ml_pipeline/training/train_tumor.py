"""
Training Pipeline — Glioblastoma Brain Tumor Detection
=======================================================
Dataset : HistologyHSI-GB
Classes : Non-Tumor Tissue (ROI_NT=0), Glioblastoma Tumor (ROI_T=1)

Pipeline (identical to inference):
  cube → radiometric correction → band selection → 32×32 grid patches
  → spectral features → spatial features → StandardScaler → PCA
  → RF + SVM → SoftmaxFusion

Saves to outputs/trained_models/:
  tumor_{rf,svm,scaler,pca}.pkl        ← canonical (used at inference)
  tumor_{rf,svm,scaler,pca}_<ts>.pkl   ← versioned backup (if ENABLE_MODEL_VERSIONING)

Saves to outputs/evaluation_metrics/:
  tumor_metrics.json                   ← consumed by GET /predict/evaluation/tumor
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import joblib
except ImportError:
    import pickle as joblib

sys.path.insert(0, str(Path(__file__).parents[2]))

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.config import (
    ENABLE_MODEL_VERSIONING, EVAL_METRICS_DIR, MAX_PATCHES, N_INFORMATIVE_BANDS,
    N_PCA_TUMOR, PATCH_SIZE, RF_MAX_DEPTH, RF_TREES, SVM_C_TUMOR, SVM_KERNEL,
    STRIDE, TRAINED_MODELS_DIR, TUMOR_DATASET_ROOT,
)
from ml_pipeline.data_loader.hsi_loader import load_hyperspectral_image
from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
from ml_pipeline.fusion.softmax_fusion import SoftmaxFusion
from ml_pipeline.preprocessing.patch_extraction import (
    extract_patches, select_informative_bands,
)
from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction, load_envi_cube

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATASET_ROOT = TUMOR_DATASET_ROOT
OUTPUT_DIR   = TRAINED_MODELS_DIR
EVAL_DIR     = EVAL_METRICS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TUMOR_CLASSES = ["Non-Tumor Tissue", "Glioblastoma Tumor"]


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_histologyhsi_dataset():
    if not DATASET_ROOT.exists():
        raise RuntimeError(
            f"Tumor dataset not found: {DATASET_ROOT}\n"
            "Set env var HSI_TUMOR_DATASET to the correct path."
        )
    return _load_real_dataset()


def _load_envi_ref(roi_dir: Path, stem: str) -> "np.ndarray | None":
    """
    Load a dark or white reference cube from an ROI folder.

    HistologyHSI-GB stores references as ENVI pairs:
      darkReference.hdr  +  darkReference  (no extension)
      whiteReference.hdr +  whiteReference (no extension)
    Returns None if the files are missing so correction gracefully
    falls back to global min-max normalisation.
    """
    hdr = roi_dir / f"{stem}.hdr"
    raw = roi_dir / stem              # no extension — ENVI convention
    if not hdr.exists() or not raw.exists():
        return None
    try:
        ref = load_hyperspectral_image(str(hdr), str(raw))
        return ref.astype(np.float32)
    except Exception as exc:
        log.debug("Reference load failed (%s/%s): %s", roi_dir.name, stem, exc)
        return None


def _load_real_dataset():
    features_list, labels_list, patient_ids = [], [], []
    subject_count = 0
    MAX_SUBJECTS  = 4

    for patient_dir in sorted(DATASET_ROOT.iterdir()):
        if subject_count >= MAX_SUBJECTS:
            break
        if not patient_dir.is_dir():
            continue

        pid = patient_dir.name
        log.info("Loading patient: %s", pid)

        for roi_dir in sorted(patient_dir.iterdir()):
            if not roi_dir.is_dir():
                continue

            roi_name = roi_dir.name.upper()
            label    = 1 if ("_T" in roi_name and "_NT" not in roi_name) else 0

            hdr_file = roi_dir / "raw.hdr"
            raw_file = roi_dir / "raw"        # no extension — ENVI convention
            if not hdr_file.exists():
                continue

            try:
                try:
                    cube = load_hyperspectral_image(str(hdr_file), str(raw_file))
                except Exception:
                    cube = load_envi_cube(str(hdr_file))

                cube = cube.astype(np.float32)

                # ── Load per-ROI dark/white references ───────────────────────
                # Each ROI folder contains darkReference(.hdr) + whiteReference(.hdr)
                # Use them for proper reflectance calibration when available;
                # otherwise apply_radiometric_correction falls back to min-max.
                dark_ref  = _load_envi_ref(roi_dir, "darkReference")
                white_ref = _load_envi_ref(roi_dir, "whiteReference")
                if dark_ref is not None and white_ref is not None:
                    log.debug("  [%s] using per-ROI dark+white refs", roi_dir.name)
                else:
                    log.debug("  [%s] refs missing — using global normalisation", roi_dir.name)

                # ── Same preprocessing as inference ──────────────────────────
                cube = apply_radiometric_correction(cube, dark_ref=dark_ref, white_ref=white_ref)
                cube = select_informative_bands(cube, n_bands=N_INFORMATIVE_BANDS)

                count = 0
                for (_, _, patch) in extract_patches(cube, PATCH_SIZE, STRIDE):
                    if count >= MAX_PATCHES:
                        break
                    # Patch is (P,P,B) — never flattened (spec §5 rule 1)
                    spec = extract_spectral_features(patch)
                    spat = extract_spatial_features(patch)
                    features_list.append(np.concatenate([spec, spat]))
                    labels_list.append(label)
                    patient_ids.append(pid)
                    count += 1

            except Exception as e:
                log.warning("Failed loading %s: %s", roi_dir.name, e)

        subject_count += 1

    return np.array(features_list), np.array(labels_list), patient_ids


def patient_wise_split(features, labels, patient_ids):
    unique    = sorted(set(patient_ids))
    split     = int(len(unique) * 0.7)
    train_set = set(unique[:split])
    test_set  = set(unique[split:])
    log.info("Train patients: %s", sorted(train_set))
    log.info("Test  patients: %s", sorted(test_set))
    tr = [i for i, p in enumerate(patient_ids) if p in train_set]
    te = [i for i, p in enumerate(patient_ids) if p in test_set]
    return features[tr], labels[tr], features[te], labels[te]


# ─── Save helpers ─────────────────────────────────────────────────────────────

def _save_models(rf, svm, scaler, pca):
    """Save canonical + versioned model files (spec §7 model versioning)."""
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    names   = {"rf": rf, "svm": svm, "scaler": scaler, "pca": pca}

    for name, model in names.items():
        canonical = OUTPUT_DIR / f"tumor_{name}.pkl"
        joblib.dump(model, canonical)
        log.info("Saved %s", canonical)

        if ENABLE_MODEL_VERSIONING:
            versioned = OUTPUT_DIR / f"tumor_{name}_{ts}.pkl"
            joblib.dump(model, versioned)
            log.info("Versioned copy → %s", versioned)


def _save_eval_metrics(y_test, rf_preds, svm_preds, fused_preds,
                        rf_probs, svm_probs, fused_probs):
    """Save evaluation JSON matching the frontend's expected structure."""
    n = len(TUMOR_CLASSES)

    def _m(y_true, y_pred, classes):
        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred,
                             average="binary" if n == 2 else "weighted",
                             zero_division=0))
        cm  = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        return {
            "accuracy":                round(acc, 4),
            "f1_score":                round(f1, 4),
            "confusion_matrix":        cm.tolist(),
            "confusion_matrix_labels": classes,
        }

    data = {
        "classes":       TUMOR_CLASSES,
        "fusion":        _m(y_test, fused_preds, TUMOR_CLASSES),
        "random_forest": _m(y_test, rf_preds,    TUMOR_CLASSES),
        "svm":           _m(y_test, svm_preds,   TUMOR_CLASSES),
    }
    out = EVAL_DIR / "tumor_metrics.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Evaluation metrics saved → %s", out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_tumor_models():
    log.info("=== Glioblastoma Tumor Detection — Training ===")
    log.info("Config: PATCH_SIZE=%d  STRIDE=%d  N_PCA=%d  RF_TREES=%d",
             PATCH_SIZE, STRIDE, N_PCA_TUMOR, RF_TREES)

    features, labels, patient_ids = load_histologyhsi_dataset()
    log.info("Dataset: %s  labels=%s", features.shape,
             dict(zip(*np.unique(labels, return_counts=True))))

    X_train, y_train, X_test, y_test = patient_wise_split(features, labels, patient_ids)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    n_comp   = min(N_PCA_TUMOR, X_train_s.shape[1], X_train_s.shape[0] - 1)
    pca      = PCA(n_components=n_comp)
    X_tr_pca = pca.fit_transform(X_train_s)
    X_te_pca = pca.transform(X_test_s)
    log.info("PCA: %d components", pca.n_components_)

    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    log.info("Training Random Forest …")
    rf.fit(X_tr_pca, y_train)

    svm = SVC(
        kernel=SVM_KERNEL, C=SVM_C_TUMOR,
        probability=True, class_weight="balanced",
    )
    log.info("Training SVM …")
    svm.fit(X_tr_pca, y_train)

    rf_preds  = rf.predict(X_te_pca)
    svm_preds = svm.predict(X_te_pca)
    rf_probs  = rf.predict_proba(X_te_pca)
    svm_probs = svm.predict_proba(X_te_pca)

    fusion      = SoftmaxFusion([rf, svm])
    fused_preds = fusion.predict(X_te_pca)
    fused_probs = fusion.predict_proba(X_te_pca)

    log.info("RF     accuracy: %.4f", accuracy_score(y_test, rf_preds))
    log.info("SVM    accuracy: %.4f", accuracy_score(y_test, svm_preds))
    log.info("Fusion accuracy: %.4f", accuracy_score(y_test, fused_preds))

    _save_models(rf, svm, scaler, pca)
    _save_eval_metrics(y_test, rf_preds, svm_preds, fused_preds,
                       rf_probs, svm_probs, fused_probs)

    # ── Research visualizations (optional, non-fatal) ───────────────────────
    try:
        from ml_pipeline.visualization.feature_importance import save_feature_importance
        n_feats = X_tr_pca.shape[1]
        feature_names = [f"pca_{i}" for i in range(n_feats)]
        save_feature_importance(rf, feature_names,
                                str(EVAL_METRICS_DIR.parent / "feature_importance" / "tumor_features.png"))
        log.info("Feature importance plot saved.")
    except Exception as exc:
        log.debug("Feature importance plot skipped: %s", exc)

    try:
        from ml_pipeline.evaluation.model_comparison import evaluate_models, plot_model_comparison
        models_dict = {"random_forest": rf, "svm": svm}
        df = evaluate_models(models_dict, X_te_pca, y_test)
        plot_model_comparison(df,
                              str(EVAL_METRICS_DIR.parent / "model_comparison" / "tumor_model_comparison.png"))
        log.info("Model comparison plot saved.")
    except Exception as exc:
        log.debug("Model comparison plot skipped: %s", exc)

    log.info("Training complete.")


if __name__ == "__main__":
    train_tumor_models()