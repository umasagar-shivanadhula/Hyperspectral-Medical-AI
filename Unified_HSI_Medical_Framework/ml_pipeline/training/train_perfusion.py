"""
Training Pipeline — Tissue Perfusion Abnormality Detection.

Dataset: SPECTRALPACA (patient-wise split)
Classes: Normal Perfusion, Reduced Perfusion, Abnormal Perfusion
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction, extract_patches
from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
from ml_pipeline.fusion.softmax_fusion import SoftmaxFusion
from ml_pipeline.evaluation.metrics import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).parents[2] / "datasets" / "perfusion_dataset" / "SPECTRALPACA"
OUTPUT_DIR = Path(__file__).parents[2] / "outputs" / "trained_models"
EVAL_DIR = Path(__file__).parents[2] / "outputs" / "evaluation_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PERFUSION_CLASSES = ["Normal Perfusion", "Reduced Perfusion", "Abnormal Perfusion"]
PATCH_SIZE = 32
N_PCA_COMPONENTS = 30
N_ESTIMATORS_RF = 200
PATIENT_WISE_SPLIT = True  # Use patient-wise split to prevent data leakage


def load_spectralpaca_dataset():
    """
    Load SPECTRALPACA dataset from disk.
    If real data not found, generates synthetic data for demonstration.
    """
    if DATASET_ROOT.exists() and any(DATASET_ROOT.iterdir()):
        return _load_real_dataset()
    else:
        log.warning("SPECTRALPACA dataset not found. Generating synthetic data.")
        return _generate_synthetic_perfusion_data()


def _load_real_dataset():
    """Load from actual SPECTRALPACA directory structure."""
    features_list = []
    labels_list = []
    patient_ids = []

    label_map = {"normal": 0, "reduced": 1, "abnormal": 2}

    for subject_dir in sorted(DATASET_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue
        pid = subject_dir.name
        log.info(f"Loading subject: {pid}")

        for phase_dir in subject_dir.iterdir():
            phase = phase_dir.name.lower()
            if "normal" in phase or "recovery" in phase:
                label = 0
            elif "clamp" in phase or "reduced" in phase:
                label = 1
            else:
                label = 2

            for cube_file in phase_dir.glob("*.npy"):
                try:
                    cube = np.load(str(cube_file)).astype(np.float32)
                    cube = apply_radiometric_correction(cube)
                    for _, _, patch in extract_patches(cube, PATCH_SIZE):
                        spec_feat = extract_spectral_features(patch)
                        spat_feat = extract_spatial_features(patch)
                        feat = np.concatenate([spec_feat, spat_feat])
                        features_list.append(feat)
                        labels_list.append(label)
                        patient_ids.append(pid)
                except Exception as e:
                    log.warning(f"Failed to load {cube_file}: {e}")

    return np.array(features_list), np.array(labels_list), patient_ids


def _generate_synthetic_perfusion_data(n_samples: int = 1500):
    """Generate synthetic HSI features for 3 perfusion classes."""
    rng = np.random.RandomState(42)
    features, labels, patients = [], [], []

    # 10 synthetic patients
    patients_list = [f"subject_{i:02d}" for i in range(1, 11)]
    class_centers = [
        rng.uniform(0.6, 0.8, 80),   # Normal: high reflectance
        rng.uniform(0.4, 0.6, 80),   # Reduced: mid reflectance
        rng.uniform(0.1, 0.3, 80),   # Abnormal: low reflectance
    ]

    for i in range(n_samples):
        label = rng.randint(0, 3)
        patient = patients_list[i % 10]
        noise = rng.randn(80) * 0.08
        feat = class_centers[label] + noise
        features.append(feat)
        labels.append(label)
        patients.append(patient)

    return np.array(features), np.array(labels), patients


def patient_wise_split(features, labels, patient_ids, test_patients=None):
    """
    Split dataset by patient to prevent spatial/temporal data leakage.
    Last 2 patients are used as test set by default.
    """
    unique_patients = sorted(set(patient_ids))
    if test_patients is None:
        test_patients = set(unique_patients[-2:])
    else:
        test_patients = set(test_patients)

    train_idx = [i for i, p in enumerate(patient_ids) if p not in test_patients]
    test_idx = [i for i, p in enumerate(patient_ids) if p in test_patients]

    log.info(f"Train patients: {[p for p in unique_patients if p not in test_patients]}")
    log.info(f"Test patients: {list(test_patients)}")
    log.info(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")

    return (features[train_idx], labels[train_idx],
            features[test_idx], labels[test_idx])


def train_perfusion_models():
    """Full training pipeline for perfusion detection."""
    log.info("=== Perfusion Model Training ===")

    # 1. Load data
    features, labels, patient_ids = load_spectralpaca_dataset()
    log.info(f"Dataset shape: {features.shape}, Classes: {np.unique(labels, return_counts=True)}")

    # 2. Patient-wise split
    X_train, y_train, X_test, y_test = patient_wise_split(features, labels, patient_ids)

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. PCA dimensionality reduction (mitigates Hughes Phenomenon)
    n_comp = min(N_PCA_COMPONENTS, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    log.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # 5. Train Random Forest (spectral branch)
    log.info("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_RF,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_pca, y_train)

    # 6. Train SVM (spatial branch)
    log.info("Training SVM classifier...")
    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    svm.fit(X_train_pca, y_train)

    # 7. Evaluate individual classifiers
    rf_preds = rf.predict(X_test_pca)
    svm_preds = svm.predict(X_test_pca)
    log.info(f"\nRandom Forest Test Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    log.info(f"\n{classification_report(y_test, rf_preds, target_names=PERFUSION_CLASSES)}")
    log.info(f"\nSVM Test Accuracy: {accuracy_score(y_test, svm_preds):.4f}")

    # 8. Fusion and evaluation
    fusion = SoftmaxFusion(classifiers=[rf, svm])
    fused_preds = fusion.predict(X_test_pca)
    rf_probs = rf.predict_proba(X_test_pca)
    svm_probs = svm.predict_proba(X_test_pca)
    fused_probs = fusion.predict_proba(X_test_pca)

    log.info(f"\nFusion Test Accuracy: {accuracy_score(y_test, fused_preds):.4f}")
    log.info(f"\n{classification_report(y_test, fused_preds, target_names=PERFUSION_CLASSES)}")

    eval_results = evaluate_classifier(
        y_test,
        rf_preds,
        svm_preds,
        fused_preds,
        PERFUSION_CLASSES,
        prob_rf=rf_probs,
        prob_svm=svm_probs,
        prob_fusion=fused_probs,
    )
    for name, metrics in eval_results.items():
        log.info(f"\n--- {name.upper()} ---")
        log.info(f"  Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score']}")
        log.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    import json
    eval_path = EVAL_DIR / "perfusion_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(
            {
                "task": "perfusion",
                "classes": PERFUSION_CLASSES,
                "random_forest": {k: v for k, v in eval_results["random_forest"].items() if k != "classification_report"},
                "svm": {k: v for k, v in eval_results["svm"].items() if k != "classification_report"},
                "fusion": {k: v for k, v in eval_results["fusion"].items() if k != "classification_report"},
            },
            f,
            indent=2,
        )
    log.info(f"Evaluation metrics saved to {eval_path}")

    # 9. Save models (joblib for sklearn compatibility)
    joblib.dump(rf, OUTPUT_DIR / "perfusion_rf.pkl")
    joblib.dump(svm, OUTPUT_DIR / "perfusion_svm.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "perfusion_scaler.pkl")
    joblib.dump(pca, OUTPUT_DIR / "perfusion_pca.pkl")

    log.info(f"Models saved to {OUTPUT_DIR}")
    return rf, svm, scaler, pca


if __name__ == "__main__":
    train_perfusion_models()
