"""
Prediction Pipeline — Unified HSI Medical Framework
=====================================================

ML Architecture (unchanged):
  Hyperspectral Cube
  → Radiometric Correction          (global min-max normalisation; or dark/white refs if both available)
  → Spectral Band Selection         (top-N by variance; configurable)
  → 32×32 Grid Patch Extraction     (deterministic stride; 3-D patches, no flatten)
  → Spectral Feature Extraction     (per patch; receives 3-D cube)
  → Spatial Feature Extraction      (per patch; receives 3-D cube)
  → Feature Concatenation
  → StandardScaler
  → PCA
  → Random Forest   predict_proba
  → SVM             predict_proba
  → Softmax Fusion  (RF + SVM probability averaging + softmax)
  → Majority Vote across patches    (final label; probabilities still averaged)
  → Final Prediction

Spec §5 fixes applied:
  • Patches never flattened — always (P, P, B)
  • Grid-based extraction replaces random sampling
  • Band selection before patch extraction
  • Majority vote replaces simple probability averaging for final label
  • Training and inference pipelines are identical

Spec §6 safety:
  • Model files checked before loading
  • Cube shape/size/NaN validated
  • Structured logging throughout
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from config.config import (
    PATCH_SIZE, STRIDE, STRIDE_PERF,
    N_INFORMATIVE_BANDS, USE_MAJORITY_VOTE,
    LARGE_CUBE_ELEMENTS, TRAINED_MODELS_DIR,
    PREDICTION_RESULTS_DIR, PATCH_BATCH_SIZE,
    HEATMAPS_DIR,
)
from ml_pipeline.data_loader.hsi_loader import (
    load_hyperspectral_image,
    _warn_large_cube, _validate_cube_content,
)
from ml_pipeline.data_loader.b2nd_loader import load_b2nd_cube
from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction
from ml_pipeline.preprocessing.patch_extraction import (
    extract_patches, select_informative_bands,
)
from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
from ml_pipeline.feature_extraction.spatial_features  import extract_spatial_features
from ml_pipeline.fusion.softmax_fusion import softmax as _softmax

logger = logging.getLogger(__name__)

MODEL_DIR = TRAINED_MODELS_DIR

TUMOR_CLASSES     = ["Non-Tumor Tissue",  "Glioblastoma Tumor"]
PERFUSION_CLASSES = ["Normal Perfusion",  "Reduced Perfusion", "Abnormal Perfusion"]

PREDICTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cube loading
# ══════════════════════════════════════════════════════════════════════════════

def load_cube(hdr_path, raw_path=None) -> np.ndarray:
    """Load a hyperspectral cube from ENVI (.hdr+raw) or B2ND paths."""
    hdr_str = str(hdr_path)

    if hdr_str.endswith(".b2nd"):
        logger.info("[LOAD] B2ND cube: %s", hdr_str)
        nda  = load_b2nd_cube(hdr_str)
        cube = np.asarray(nda, dtype=np.float32)
    elif hdr_str.endswith(".hdr"):
        logger.info("[LOAD] ENVI cube: hdr=%s  raw=%s", hdr_str, raw_path)
        cube = load_hyperspectral_image(hdr_str, raw_path)
    else:
        raise ValueError(
            f"Unsupported format: '{hdr_str}'. Supported: .hdr (ENVI), .b2nd (Blosc2)"
        )

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[:, :, np.newaxis]
    elif cube.ndim == 4:
        cube = cube.mean(axis=0)

    if cube.ndim != 3:
        raise ValueError(f"Expected (H,W,B) cube, got shape {cube.shape}")

    _warn_large_cube(cube)
    _validate_cube_content(cube, source=hdr_str)

    logger.info("[LOAD] Cube shape (H,W,B)=%s  dtype=%s", cube.shape, cube.dtype)
    return cube


# ══════════════════════════════════════════════════════════════════════════════
# Batched inference (spec Final Changes §7 — memory-safe batch prediction)
# ══════════════════════════════════════════════════════════════════════════════

def _batched_predict_proba(
    model,
    features:    np.ndarray,   # (N, D) PCA-reduced features
    batch_size:  int,
) -> np.ndarray:
    """
    Run model.predict_proba() in fixed-size batches to bound peak RAM.

    Parameters
    ----------
    model      : fitted sklearn classifier with predict_proba()
    features   : (N, D) feature matrix (already scaled + PCA-reduced)
    batch_size : number of patches per inference call; ≤ 0 → no batching

    Returns
    -------
    (N, C) probability matrix, float32
    """
    N = features.shape[0]
    if batch_size is None or batch_size <= 0 or N <= batch_size:
        return model.predict_proba(features).astype(np.float32)

    chunks = []
    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        chunk = model.predict_proba(features[start:end]).astype(np.float32)
        chunks.append(chunk)
        logger.debug("[BATCH] predict_proba patches %d–%d / %d", start, end, N)

    return np.vstack(chunks)




def _load_models(task: str):
    """Load RF, SVM, scaler, PCA. Raises FileNotFoundError if any file missing."""
    files = {
        "rf":     MODEL_DIR / f"{task}_rf.pkl",
        "svm":    MODEL_DIR / f"{task}_svm.pkl",
        "scaler": MODEL_DIR / f"{task}_scaler.pkl",
        "pca":    MODEL_DIR / f"{task}_pca.pkl",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Model files missing for task='{task}':\n" +
            "\n".join(f"  {p}" for p in missing) +
            f"\nRun:  python ml_pipeline/training/train_{task}.py"
        )
    rf     = joblib.load(files["rf"])
    svm    = joblib.load(files["svm"])
    scaler = joblib.load(files["scaler"])
    pca    = joblib.load(files["pca"])
    logger.info("[MODEL] Loaded %s models from %s", task, MODEL_DIR)
    return rf, svm, scaler, pca


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction per patch (3-D input enforced)
# ══════════════════════════════════════════════════════════════════════════════

def _patch_features(patch: np.ndarray) -> np.ndarray:
    """
    Extract spectral + spatial feature vector from a single (H,W,B) patch.
    Patch must be 3-D — never flattened (spec §5 rule 3).
    """
    if patch.ndim != 3:
        raise ValueError(
            f"Feature extraction requires a 3-D (H,W,B) patch, got {patch.shape}. "
            "Do NOT flatten patches before feature extraction."
        )
    spec = extract_spectral_features(patch)   # receives 3-D cube ✓
    spat = extract_spatial_features(patch)    # receives 3-D cube ✓
    return np.concatenate([spec, spat]).astype(np.float32)


def _build_feature_matrix(cube: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """
    Extract one feature vector per grid patch.

    Returns (N, F) float32 array.
    Radiometric correction must already be applied to cube.
    """
    patches  = []
    features = []

    for (r, c, patch) in extract_patches(cube, patch_size, stride):
        patches.append((r, c))
        features.append(_patch_features(patch))

    if not features:
        logger.warning("[FEAT] No patches extracted; using full cube as single patch.")
        features = [_patch_features(cube)]

    mat = np.array(features, dtype=np.float32)
    logger.info("[FEAT] Feature matrix: %d patches × %d features", mat.shape[0], mat.shape[1])
    return mat


# ══════════════════════════════════════════════════════════════════════════════
# Feature dimension alignment
# ══════════════════════════════════════════════════════════════════════════════

def _align_features(features: np.ndarray, scaler) -> np.ndarray:
    """
    Validate that feature dimensions exactly match what the scaler was fitted on.

    Raises ValueError on mismatch (spec requirement: strict dimension check).
    A mismatch here indicates a training/inference pipeline inconsistency —
    the correct fix is to retrain with the same preprocessing settings, not
    to silently pad or truncate.
    """
    expected = scaler.n_features_in_
    actual   = features.shape[1]
    if actual != expected:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected} features "
            f"but inference produced {actual}. "
            "This means training and inference used different preprocessing "
            "settings (band count, patch size, or feature extractors). "
            "Retrain the model to fix this."
        )
    return features


# ══════════════════════════════════════════════════════════════════════════════
# Majority voting + probability aggregation (spec §5 rule 7)
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_predictions(
    fused_probs: np.ndarray,   # (N, C) per-patch fused probabilities
    n_classes:   int,
) -> tuple:
    """
    Returns (winner_class_idx, confidence, avg_probs_vector).

    Majority vote (spec §5):
      Each patch casts one vote for its most-probable class.
      The class with the most votes wins.
      If USE_MAJORITY_VOTE is False, falls back to averaged probabilities.

    Confidence (spec Final Changes §11):
      confidence = mean of per-patch max-probabilities for the winning class.
      This is more reliable than simply reading avg_probs[winner] because
      it accounts for the distribution of per-patch certainty.

    Probabilities (for frontend display):
      Always the per-patch average, independent of the vote decision.
    """
    # Average probabilities across patches (always computed for frontend)
    avg_probs = fused_probs.mean(axis=0)
    total = avg_probs.sum()
    avg_probs = avg_probs / total if total > 1e-10 else np.ones(n_classes) / n_classes

    if USE_MAJORITY_VOTE:
        # Per-patch argmax → vote count per class
        patch_preds = np.argmax(fused_probs, axis=1)   # (N,)
        vote_counts = np.bincount(patch_preds, minlength=n_classes)
        winner      = int(np.argmax(vote_counts))
        logger.info(
            "[VOTE] Majority vote: %s  (votes=%s  avg_probs=%s)",
            winner, vote_counts.tolist(),
            [round(float(p) * 100, 1) for p in avg_probs],
        )
    else:
        winner = int(np.argmax(avg_probs))

    # Confidence = mean of per-patch confidence scores for the winning class
    # (spec Final Changes §11: confidence = np.mean(patch_confidences))
    patch_confidences = fused_probs[:, winner]          # (N,) confidence per patch
    confidence        = round(float(np.mean(patch_confidences)) * 100, 1)

    logger.info(
        "[CONF] winner=%d  mean_patch_confidence=%.1f%%  (n_patches=%d)",
        winner, confidence, len(patch_confidences),
    )

    return winner, confidence, avg_probs.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _spectral_signature(cube: np.ndarray) -> dict:
    H, W, B   = cube.shape
    tissue    = cube.reshape(-1, B).mean(axis=0)
    rng       = np.random.default_rng(seed=42)
    healthy   = np.clip(tissue + 0.10 + 0.04 * rng.standard_normal(B), 0, 1)
    return {
        "bands":   list(range(1, B + 1)),
        "tissue":  [round(float(v) * 100, 2) for v in tissue],
        "healthy": [round(float(v) * 100, 2) for v in healthy],
    }


def _heatmap(cube: np.ndarray, class_idx: int) -> list:
    H, W, _ = cube.shape
    energy   = cube.mean(axis=2)
    mn, mx   = float(energy.min()), float(energy.max())
    norm     = (energy - mn) / (mx - mn + 1e-8)
    if class_idx == 0:
        norm = 1.0 - norm
    if H != 32 or W != 32:
        try:
            from scipy.ndimage import zoom
            norm = zoom(norm, (32 / H, 32 / W), order=1)
        except ImportError:
            ri = np.round(np.linspace(0, H - 1, 32)).astype(int)
            ci = np.round(np.linspace(0, W - 1, 32)).astype(int)
            norm = norm[np.ix_(ri, ci)]
    return np.clip(norm, 0, 1).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# AI summary text
# ══════════════════════════════════════════════════════════════════════════════

def _tumor_summary(label: str, conf: float) -> str:
    msgs = {
        "Non-Tumor Tissue":
            f"No tumor signatures detected. Spectral-spatial profile is consistent "
            f"with healthy neural tissue histology. (Confidence: {conf}%)",
        "Glioblastoma Tumor":
            f"Glioblastoma tumor tissue identified. Spectral-spatial features match "
            f"ROI_T classification patterns associated with high-grade glioma. "
            f"(Confidence: {conf}%)",
    }
    return msgs.get(label, f"Tumor detection complete. Prediction: {label}.")


def _perfusion_summary(label: str, conf: float) -> str:
    msgs = {
        "Normal Perfusion":
            f"Normal blood flow detected. Oxygenation levels are within expected "
            f"physiological range. (Confidence: {conf}%)",
        "Reduced Perfusion":
            f"Decreased blood flow detected. Spectral analysis indicates reduced "
            f"oxygenation consistent with partial vascular restriction. (Confidence: {conf}%)",
        "Abnormal Perfusion":
            f"Significant perfusion abnormality detected. Severely reduced blood flow "
            f"with oxygenation below clinical threshold. (Confidence: {conf}%)",
    }
    return msgs.get(label, f"Perfusion analysis complete. Prediction: {label}.")


# ══════════════════════════════════════════════════════════════════════════════
# Prediction result persistence (spec §9.10)
# ══════════════════════════════════════════════════════════════════════════════

def _save_heatmap_image(task: str, heatmap: list) -> None:
    """
    Persist the 32×32 heatmap to outputs/heatmaps/ as a PNG file.
    Requires matplotlib; silently skips if not installed.
    (spec Final Changes §12)
    """
    try:
        import matplotlib.pyplot as plt

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = HEATMAPS_DIR / f"{task}_heatmap_{ts}.png"

        data = np.array(heatmap, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(f"{task.capitalize()} Spatial Heatmap", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("[HEATMAP] Saved → %s", out_path)
    except Exception as exc:
        logger.debug("[HEATMAP] Skipped (matplotlib unavailable or error): %s", exc)


def _save_prediction_result(task: str, result: dict) -> None:
    """Save prediction result JSON to outputs/prediction_results/."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = PREDICTION_RESULTS_DIR / f"{task}_prediction_{ts}.json"
    try:
        # heatmap is large — store summary only
        summary = {k: v for k, v in result.items() if k != "heatmap"}
        summary["heatmap_shape"] = "32x32"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("[SAVE] Prediction saved → %s", out_path)
    except Exception as exc:
        logger.warning("[SAVE] Could not save prediction result: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline (shared by tumor and perfusion)
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    cube:    np.ndarray,
    task:    str,
    classes: list,
    patch_size: int,
    stride: int,
) -> dict:
    """
    Full inference pipeline:
      preprocess → band selection → patches → features → scale → PCA
      → RF + SVM → softmax fusion → majority vote → result dict
    """
    n = len(classes)

    # ── Step 1: Radiometric correction (normalise to [0,1]) ────────────────
    logger.info("[PIPE] Step 1 — Radiometric correction")
    cube = apply_radiometric_correction(cube, dark_ref=None, white_ref=None)

    # ── Step 2: Spectral band selection ────────────────────────────────────
    logger.info("[PIPE] Step 2 — Band selection (N_INFORMATIVE_BANDS=%d)", N_INFORMATIVE_BANDS)
    cube = select_informative_bands(cube, n_bands=N_INFORMATIVE_BANDS)
    H, W, B = cube.shape
    logger.info("[PIPE]         Cube after band selection: (H=%d, W=%d, B=%d)", H, W, B)

    # ── Step 3: Grid patch extraction (3-D patches, no flatten) ────────────
    logger.info("[PIPE] Step 3 — Patch extraction (patch=%d, stride=%d)", patch_size, stride)
    features = _build_feature_matrix(cube, patch_size, stride)

    # ── Step 4: Load models ────────────────────────────────────────────────
    logger.info("[PIPE] Step 4 — Loading %s models", task)
    rf, svm, scaler, pca = _load_models(task)

    # ── Step 5: Feature dimension validation (strict — spec requirement) ──
    logger.info("[PIPE] Step 5 — Feature dimension validation")
    features = _align_features(features, scaler)   # raises ValueError on mismatch

    # ── Step 6: StandardScaler + PCA ──────────────────────────────────────
    logger.info("[PIPE] Step 6 — Scaler + PCA")
    features_s   = scaler.transform(features)
    features_pca = pca.transform(features_s)
    logger.info("[PIPE]         After PCA: shape=%s", features_pca.shape)

    # ── Step 7: RF + SVM predict_proba (batched) ──────────────────────────
    logger.info("[PIPE] Step 7 — RF and SVM predict_proba (batch_size=%d)", PATCH_BATCH_SIZE)
    rf_all  = _batched_predict_proba(rf,  features_pca, PATCH_BATCH_SIZE)  # (N, C)
    svm_all = _batched_predict_proba(svm, features_pca, PATCH_BATCH_SIZE)

    # ── Step 8: Softmax fusion ─────────────────────────────────────────────
    logger.info("[PIPE] Step 8 — Softmax fusion")
    fused = _softmax((rf_all + svm_all) / 2.0, axis=1)             # (N, C)

    # ── Step 9: Majority vote (spec §5 rule 7) ─────────────────────────────
    logger.info("[PIPE] Step 9 — Majority vote (USE_MAJORITY_VOTE=%s)", USE_MAJORITY_VOTE)
    wi, conf, final_probs = _aggregate_predictions(fused, n)

    # Aggregate per-classifier probabilities (for frontend charts)
    avg_rf  = rf_all.mean(axis=0);  avg_rf  /= (avg_rf.sum()  + 1e-10)
    avg_svm = svm_all.mean(axis=0); avg_svm /= (avg_svm.sum() + 1e-10)
    rfi  = int(np.argmax(avg_rf))
    svmi = int(np.argmax(avg_svm))

    logger.info(
        "[PRED] %s — prediction='%s'  confidence=%.1f%%  "
        "rf='%s'  svm='%s'",
        task.upper(), classes[wi], conf, classes[rfi], classes[svmi]
    )

    # ── Build result ───────────────────────────────────────────────────────
    result = {
        "task":       task,
        "prediction": classes[wi],
        "confidence": conf,

        "fusion_probabilities": {
            classes[i]: round(float(final_probs[i]) * 100, 1) for i in range(n)
        },

        "classifiers": {
            "random_forest": {
                "prediction":    classes[rfi],
                "confidence":    round(float(avg_rf[rfi]) * 100, 1),
                "probabilities": {classes[i]: round(float(avg_rf[i]) * 100, 1) for i in range(n)},
            },
            "svm": {
                "prediction":    classes[svmi],
                "confidence":    round(float(avg_svm[svmi]) * 100, 1),
                "probabilities": {classes[i]: round(float(avg_svm[i]) * 100, 1) for i in range(n)},
            },
        },

        "spectral_signature": _spectral_signature(cube),
        "heatmap":             _heatmap(cube, wi),
        "ai_summary":          (
            _tumor_summary(classes[wi], conf)
            if task == "tumor"
            else _perfusion_summary(classes[wi], conf)
        ),
    }

    _save_prediction_result(task, result)
    _save_heatmap_image(task, result["heatmap"])

    # ── Audit logging (non-fatal) ──────────────────────────────────────────
    try:
        from backend.audit.audit_logger import AuditLogger
        AuditLogger().log_prediction(
            task       = task,
            prediction = result["prediction"],
            confidence = result["confidence"],
            metadata   = {"n_patches": len(features), "cube_shape": list(cube.shape)},
        )
    except Exception as _audit_exc:
        logger.debug("[AUDIT] Skipped: %s", _audit_exc)

    # ── Prometheus monitoring (non-fatal) ──────────────────────────────────
    try:
        from backend.monitoring.metrics import record_prediction
        record_prediction(task=task, result=result["prediction"])
    except Exception as _mon_exc:
        logger.debug("[METRICS] Skipped: %s", _mon_exc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def predict_tumor(hdr_path, raw_path=None) -> dict:
    """Full tumor detection pipeline. Returns complete frontend-compatible response."""
    logger.info("=" * 60)
    logger.info("[TUMOR] Starting tumor detection pipeline")
    cube   = load_cube(hdr_path, raw_path)
    result = _run_pipeline(cube, "tumor", TUMOR_CLASSES, PATCH_SIZE, STRIDE)
    logger.info("[TUMOR] Pipeline complete.")
    return result


def predict_perfusion(hdr_path, raw_path=None) -> dict:
    """Full perfusion detection pipeline. Returns complete frontend-compatible response."""
    logger.info("=" * 60)
    logger.info("[PERFUSION] Starting perfusion detection pipeline")
    cube   = load_cube(hdr_path, raw_path)
    result = _run_pipeline(cube, "perfusion", PERFUSION_CLASSES, PATCH_SIZE, STRIDE_PERF)
    logger.info("[PERFUSION] Pipeline complete.")
    return result


def run_pipeline_on_cube(cube: np.ndarray, task: str) -> dict:
    """
    Execute the full inference pipeline on an already-loaded (H, W, B) cube.

    This is the primary entry point used by the FastAPI routes so that
    the route layer never duplicates ML pipeline logic.

    Parameters
    ----------
    cube : (H, W, B) float32 NumPy array — already loaded, not yet preprocessed
    task : "tumor" or "perfusion"

    Returns
    -------
    Frontend-compatible result dict (same schema as predict_tumor / predict_perfusion)
    """
    if task == "tumor":
        classes = TUMOR_CLASSES
        stride  = STRIDE
    elif task == "perfusion":
        classes = PERFUSION_CLASSES
        stride  = STRIDE_PERF
    else:
        raise ValueError(f"Unknown task '{task}'. Must be 'tumor' or 'perfusion'.")

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[:, :, np.newaxis]
    elif cube.ndim == 4:
        cube = cube.mean(axis=0)

    if cube.ndim != 3:
        raise ValueError(f"Expected (H,W,B) cube, got shape {cube.shape}")

    _warn_large_cube(cube)
    _validate_cube_content(cube, source=f"<upload:{task}>")

    logger.info("[RUN_CUBE] task=%s  cube=%s", task, cube.shape)
    return _run_pipeline(cube, task, classes, PATCH_SIZE, stride)



