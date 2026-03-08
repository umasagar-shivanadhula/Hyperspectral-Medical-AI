"""
Training Pipeline — Tissue Perfusion Abnormality Detection
==========================================================
Dataset : SPECTRALPACA
Classes : Normal Perfusion (0), Reduced Perfusion (1), Abnormal Perfusion (2)

Pipeline (identical to inference):
  cube → radiometric correction → band selection → 32×32 grid patches
  → spectral features → spatial features → StandardScaler → PCA
  → RF + SVM → SoftmaxFusion

Saves to outputs/trained_models/:
  perfusion_{rf,svm,scaler,pca}.pkl        ← canonical
  perfusion_{rf,svm,scaler,pca}_<ts>.pkl   ← versioned backup

Saves to outputs/evaluation_metrics/:
  perfusion_metrics.json                   ← consumed by GET /predict/evaluation/perfusion
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
    N_PCA_PERF, PATCH_SIZE, RF_MAX_DEPTH, RF_TREES, SVM_C_PERF, SVM_KERNEL,
    STRIDE_PERF, TRAINED_MODELS_DIR, PERFUSION_DATASET_ROOT,
)
from ml_pipeline.data_loader.b2nd_loader import load_b2nd_cube
from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
from ml_pipeline.fusion.softmax_fusion import SoftmaxFusion
from ml_pipeline.preprocessing.patch_extraction import (
    extract_patches, select_informative_bands,
)
from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATASET_ROOT = PERFUSION_DATASET_ROOT
OUTPUT_DIR   = TRAINED_MODELS_DIR
EVAL_DIR     = EVAL_METRICS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PERFUSION_CLASSES = ["Normal Perfusion", "Reduced Perfusion", "Abnormal Perfusion"]


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_spectralpaca_dataset():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}\n"
            "Set env var HSI_DATASETS_ROOT to the datasets root directory."
        )
    return _load_real_dataset()


def _load_real_dataset():
    features_list, labels_list, patient_ids = [], [], []
    count = 0

    dark_ref  = None
    white_ref = None

    dark_p  = DATASET_ROOT / "dark.b2nd"
    white_p = DATASET_ROOT / "white.b2nd"
    if dark_p.exists():
        log.info("Loading dark reference …")
        dark_ref = load_b2nd_cube(dark_p)
    if white_p.exists():
        log.info("Loading white reference …")
        white_ref = load_b2nd_cube(white_p)

    for entry in sorted(DATASET_ROOT.glob("subject_*.b2nd"))[:10]:
        log.info("Processing %s", entry.name)
        pid = entry.stem
        nda = load_b2nd_cube(entry)

        for i in range(0, nda.shape[0], 100):
            if count >= MAX_PATCHES:
                break
            if i % 100 == 0:
                log.info("  slice %d / %d", i, nda.shape[0])

            slice_cube = np.asarray(nda[i], dtype=np.float32)

            # ── Same preprocessing as inference ──────────────────────────────
            slice_cube = apply_radiometric_correction(
                slice_cube, dark_ref=dark_ref, white_ref=white_ref
            )
            slice_cube = select_informative_bands(slice_cube, n_bands=N_INFORMATIVE_BANDS)

            for (_, _, patch) in extract_patches(slice_cube, PATCH_SIZE, STRIDE_PERF):
                if count >= MAX_PATCHES:
                    break
                # Patch is (P,P,B) — never flattened (spec §5 rule 1)
                spec = extract_spectral_features(patch)
                spat = extract_spatial_features(patch)
                feat = np.concatenate([spec, spat]).astype(np.float32)

                mean_i = float(np.mean(patch))
                label  = 0 if mean_i > 0.55 else (1 if mean_i > 0.35 else 2)

                features_list.append(feat)
                labels_list.append(label)
                patient_ids.append(pid)
                count += 1

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    log.info("Total samples: %s  labels=%s", X.shape,
             dict(zip(*np.unique(y, return_counts=True))))
    return X, y, patient_ids


def patient_wise_split(features, labels, patient_ids):
    unique   = sorted(set(patient_ids))
    test_set = set(unique[-3:])
    tr = [i for i, p in enumerate(patient_ids) if p not in test_set]
    te = [i for i, p in enumerate(patient_ids) if p in test_set]
    log.info("Train: %s  Test: %s", sorted(set(patient_ids) - test_set), sorted(test_set))
    return features[tr], labels[tr], features[te], labels[te]


# ─── Save helpers ─────────────────────────────────────────────────────────────

def _save_models(rf, svm, scaler, pca):
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = {"rf": rf, "svm": svm, "scaler": scaler, "pca": pca}
    for name, model in names.items():
        canonical = OUTPUT_DIR / f"perfusion_{name}.pkl"
        joblib.dump(model, canonical)
        log.info("Saved %s", canonical)
        if ENABLE_MODEL_VERSIONING:
            versioned = OUTPUT_DIR / f"perfusion_{name}_{ts}.pkl"
            joblib.dump(model, versioned)
            log.info("Versioned copy → %s", versioned)


def _save_eval_metrics(y_test, rf_preds, svm_preds, fused_preds,
                        rf_probs, svm_probs, fused_probs):
    n = len(PERFUSION_CLASSES)

    def _m(y_true, y_pred, classes):
        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm  = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        return {
            "accuracy":                round(acc, 4),
            "f1_score":                round(f1, 4),
            "confusion_matrix":        cm.tolist(),
            "confusion_matrix_labels": classes,
        }

    data = {
        "classes":       PERFUSION_CLASSES,
        "fusion":        _m(y_test, fused_preds, PERFUSION_CLASSES),
        "random_forest": _m(y_test, rf_preds,    PERFUSION_CLASSES),
        "svm":           _m(y_test, svm_preds,   PERFUSION_CLASSES),
    }
    out = EVAL_DIR / "perfusion_metrics.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Evaluation metrics saved → %s", out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def train_perfusion_models():
    log.info("=== Perfusion Abnormality Detection — Training ===")
    log.info("Config: PATCH_SIZE=%d  STRIDE=%d  N_PCA=%d  RF_TREES=%d",
             PATCH_SIZE, STRIDE_PERF, N_PCA_PERF, RF_TREES)

    X, y, patient_ids = load_spectralpaca_dataset()

    X_train, y_train, X_test, y_test = patient_wise_split(X, y, patient_ids)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    n_comp   = min(N_PCA_PERF, X_train_s.shape[1], X_train_s.shape[0] - 1)
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
        kernel=SVM_KERNEL, C=SVM_C_PERF,
        probability=True, class_weight="balanced", random_state=42,
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
                                str(EVAL_METRICS_DIR.parent / "feature_importance" / "perfusion_features.png"))
        log.info("Feature importance plot saved.")
    except Exception as exc:
        log.debug("Feature importance plot skipped: %s", exc)

    try:
        from ml_pipeline.evaluation.model_comparison import evaluate_models, plot_model_comparison
        models_dict = {"random_forest": rf, "svm": svm}
        df = evaluate_models(models_dict, X_te_pca, y_test)
        plot_model_comparison(df,
                              str(EVAL_METRICS_DIR.parent / "model_comparison" / "perfusion_model_comparison.png"))
        log.info("Model comparison plot saved.")
    except Exception as exc:
        log.debug("Model comparison plot skipped: %s", exc)

    log.info("Training complete.")


if __name__ == "__main__":
    train_perfusion_models()
