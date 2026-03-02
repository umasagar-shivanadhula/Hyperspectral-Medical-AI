"""
Training Pipeline — Glioblastoma Brain Tumor Detection.

Dataset: HistologyHSI-GB (patient-wise split)
Classes: Non-Tumor Tissue (ROI_NT), Glioblastoma Tumor (ROI_T)
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path

try:
    import joblib
except ImportError:
    import pickle as joblib

sys.path.insert(0, str(Path(__file__).parents[2]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from ml_pipeline.preprocessing.radiometric import (
    apply_radiometric_correction,
    extract_patches,
    load_envi_cube,
)
from ml_pipeline.data_loader.hsi_loader import load_hyperspectral_image
from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
from ml_pipeline.fusion.softmax_fusion import SoftmaxFusion
from ml_pipeline.evaluation.metrics import evaluate_classifier, compute_metrics_per_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Use external dataset location on E: drive
DATASET_ROOT = Path(r"E:\umasagar\datasets\tumor_dataset\HistologyHSI-GB")
OUTPUT_DIR = Path(__file__).parents[2] / "outputs" / "trained_models"
EVAL_DIR = Path(__file__).parents[2] / "outputs" / "evaluation_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TUMOR_CLASSES = ["Non-Tumor Tissue", "Glioblastoma Tumor"]
PATCH_SIZE = 32
N_PCA_COMPONENTS = 25
N_ESTIMATORS_RF = 300
STRIDE = 16


def load_histologyhsi_dataset():
    """
    Load HistologyHSI-GB dataset.
    Falls back to synthetic data if real dataset not found.
    """
    if DATASET_ROOT.exists() and any(DATASET_ROOT.iterdir()):
        return _load_real_tumor_dataset()
    else:
        log.warning("HistologyHSI-GB dataset not found. Generating synthetic data.")
        return _generate_synthetic_tumor_data()


def _load_real_tumor_dataset():
    """
    Load from HistologyHSI-GB directory structure:
    P{N}/ROI_{idx}_C{idx}_{T|NT}/raw, raw.hdr, darkReference, whiteReference, rgb.png
    """
    features_list, labels_list, patient_ids = [], [], []

    for patient_dir in sorted(DATASET_ROOT.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        log.info(f"Loading patient: {pid}")

        for roi_dir in patient_dir.iterdir():
            if not roi_dir.is_dir():
                continue

            # Determine tumor vs non-tumor from directory name
            roi_name = roi_dir.name.upper()
            if "_T" in roi_name and "_NT" not in roi_name:
                label = 1  # Tumor
            else:
                label = 0  # Non-tumor

            hdr_file = roi_dir / "raw.hdr"
            raw_file = roi_dir / "raw"
            dark_file = roi_dir / "darkReference"
            white_file = roi_dir / "whiteReference"

            if not hdr_file.exists():
                continue

            try:
                try:
                    cube = load_hyperspectral_image(str(hdr_file))
                except Exception:
                    cube = load_envi_cube(str(hdr_file))

                # Apply radiometric correction if references exist
                dark = None
                white = None
                if dark_file.exists():
                    dark = np.fromfile(str(dark_file), dtype=np.float32)
                    if dark.size == cube.shape[2]:
                        dark = dark.reshape(cube.shape[2])
                if white_file.exists():
                    white = np.fromfile(str(white_file), dtype=np.float32)
                    if white.size == cube.shape[2]:
                        white = white.reshape(cube.shape[2])
                cube = apply_radiometric_correction(cube, dark, white)

                for _, _, patch in extract_patches(cube, PATCH_SIZE, STRIDE):
                    spec_feat = extract_spectral_features(patch)
                    spat_feat = extract_spatial_features(patch)
                    feat = np.concatenate([spec_feat, spat_feat])
                    features_list.append(feat)
                    labels_list.append(label)
                    patient_ids.append(pid)

            except Exception as e:
                log.warning(f"Failed to load ROI {roi_dir.name}: {e}")

    return np.array(features_list), np.array(labels_list), patient_ids


def _generate_synthetic_tumor_data(n_samples: int = 1200):
    """
    Generate synthetic HSI features for tumor binary classification.
    Tumor tissue has distinct spectral characteristics from normal tissue.
    """
    rng = np.random.RandomState(123)
    features, labels, patients = [], [], []

    # 13 synthetic patients (matching HistologyHSI-GB)
    patient_list = [f"P{i}" for i in range(1, 14)]

    # Class-specific spectral signatures
    normal_center = rng.uniform(0.55, 0.75, 80)
    tumor_center = rng.uniform(0.25, 0.45, 80)
    # Tumor shows different spectral absorption features
    tumor_center[20:30] *= 0.6  # absorption in specific bands

    for i in range(n_samples):
        label = rng.randint(0, 2)
        patient = patient_list[i % 13]
        noise_scale = 0.07
        center = tumor_center if label == 1 else normal_center
        noise = rng.randn(80) * noise_scale
        # Add patient-specific variability
        patient_effect = rng.randn(80) * 0.02
        feat = center + noise + patient_effect
        features.append(feat)
        labels.append(label)
        patients.append(patient)

    return np.array(features), np.array(labels), patients


def patient_wise_split(features, labels, patient_ids, n_test_patients: int = 3):
    """Patient-wise split to avoid data leakage between train and test."""
    unique_patients = sorted(set(patient_ids))
    test_patients = set(unique_patients[-n_test_patients:])

    train_idx = [i for i, p in enumerate(patient_ids) if p not in test_patients]
    test_idx = [i for i, p in enumerate(patient_ids) if p in test_patients]

    log.info(f"Test patients: {list(test_patients)} ({len(test_idx)} samples)")
    log.info(f"Train samples: {len(train_idx)}")

    return (features[train_idx], labels[train_idx],
            features[test_idx], labels[test_idx])


def train_tumor_models():
    """Full training pipeline for glioblastoma detection."""
    log.info("=== Glioblastoma Tumor Detection Model Training ===")

    # 1. Load dataset
    features, labels, patient_ids = load_histologyhsi_dataset()
    log.info(f"Dataset shape: {features.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        log.info(f"  {TUMOR_CLASSES[u]}: {c} samples")

    # 2. Patient-wise split
    X_train, y_train, X_test, y_test = patient_wise_split(features, labels, patient_ids)

    # 3. Feature scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. PCA reduction (mitigates Hughes Phenomenon)
    n_comp = min(N_PCA_COMPONENTS, X_train_s.shape[1], X_train_s.shape[0] - 1)
    pca = PCA(n_components=n_comp, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    log.info(f"PCA: {n_comp} components, {pca.explained_variance_ratio_.sum():.3f} variance explained")

    # 5. Train Random Forest
    log.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF,
        max_depth=20,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_pca, y_train)

    # 6. Train SVM
    log.info("Training SVM...")
    svm = SVC(
        kernel="rbf",
        C=50.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    svm.fit(X_train_pca, y_train)

    # 7. Evaluate
    rf_preds = rf.predict(X_test_pca)
    svm_preds = svm.predict(X_test_pca)
    log.info(f"\nRandom Forest:")
    log.info(f"  Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    log.info(f"\n{classification_report(y_test, rf_preds, target_names=TUMOR_CLASSES)}")

    # AUC score
    rf_probs = rf.predict_proba(X_test_pca)[:, 1]
    try:
        auc = roc_auc_score(y_test, rf_probs)
        log.info(f"  RF AUC-ROC: {auc:.4f}")
    except Exception:
        pass

    # 8. Fusion and evaluation
    fusion = SoftmaxFusion(classifiers=[rf, svm])
    fused_preds = fusion.predict(X_test_pca)
    rf_probs = rf.predict_proba(X_test_pca)
    svm_probs = svm.predict_proba(X_test_pca)
    fused_probs = fusion.predict_proba(X_test_pca)

    log.info(f"\nFusion Accuracy: {accuracy_score(y_test, fused_preds):.4f}")
    log.info(f"\n{classification_report(y_test, fused_preds, target_names=TUMOR_CLASSES)}")

    eval_results = evaluate_classifier(
        y_test,
        rf_preds,
        svm_preds,
        fused_preds,
        TUMOR_CLASSES,
        prob_rf=rf_probs,
        prob_svm=svm_probs,
        prob_fusion=fused_probs,
    )
    for name, metrics in eval_results.items():
        log.info(f"\n--- {name.upper()} ---")
        log.info(f"  Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score']}")
        log.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    import json
    eval_path = EVAL_DIR / "tumor_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(
            {
                "task": "tumor",
                "classes": TUMOR_CLASSES,
                "random_forest": {k: v for k, v in eval_results["random_forest"].items() if k != "classification_report"},
                "svm": {k: v for k, v in eval_results["svm"].items() if k != "classification_report"},
                "fusion": {k: v for k, v in eval_results["fusion"].items() if k != "classification_report"},
            },
            f,
            indent=2,
        )
    log.info(f"Evaluation metrics saved to {eval_path}")

    # 9. Save models (joblib for sklearn compatibility)
    joblib.dump(rf, OUTPUT_DIR / "tumor_rf.pkl")
    joblib.dump(svm, OUTPUT_DIR / "tumor_svm.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "tumor_scaler.pkl")
    joblib.dump(pca, OUTPUT_DIR / "tumor_pca.pkl")

    log.info(f"\nModels saved to {OUTPUT_DIR}")
    return rf, svm, scaler, pca


if __name__ == "__main__":
    train_tumor_models()
