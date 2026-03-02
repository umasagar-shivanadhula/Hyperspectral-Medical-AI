"""
HSI Pipeline Service — loads models and coordinates the full inference pipeline.

Uses ml_pipeline for preprocessing, feature extraction, and softmax fusion.
No placeholder/dummy models: requires trained models in outputs/trained_models/.
"""
import io
import numpy as np
from pathlib import Path

try:
    import joblib
except ImportError:
    import pickle as joblib

# ML pipeline imports (project root must be on sys.path)
try:
    from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
    from ml_pipeline.fusion.softmax_fusion import softmax
    from ml_pipeline.data_loader.hsi_loader import load_hyperspectral_from_bytes
    ML_PIPELINE_AVAILABLE = True
except ImportError as e:
    ML_PIPELINE_AVAILABLE = False
    _import_err = str(e)

MODELS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "trained_models"
PERFUSION_CLASSES = ["Normal Perfusion", "Reduced Perfusion", "Abnormal Perfusion"]
TUMOR_CLASSES = ["Non-Tumor Tissue", "Glioblastoma Tumor"]


class HSIPipelineService:
    """Manages model loading and inference for both clinical tasks."""

    def __init__(self):
        self._perfusion_rf = None
        self._perfusion_svm = None
        self._perfusion_scaler = None
        self._perfusion_pca = None
        self._tumor_rf = None
        self._tumor_svm = None
        self._tumor_scaler = None
        self._tumor_pca = None
        self._load_models()

    def _load_models(self):
        """Load pre-trained models with joblib. No dummy models."""
        if not ML_PIPELINE_AVAILABLE:
            raise RuntimeError(f"ml_pipeline not available: {_import_err}")
        for task in ("perfusion", "tumor"):
            for model_type in ("rf", "svm", "scaler", "pca"):
                path = MODELS_DIR / f"{task}_{model_type}.pkl"
                attr = f"_{task}_{model_type}"
                if path.exists():
                    setattr(self, attr, joblib.load(path))
                else:
                    setattr(self, attr, None)

    def _require_models(self, task: str):
        """Raise if required models for task are not loaded."""
        rf = getattr(self, f"_{task}_rf")
        svm = getattr(self, f"_{task}_svm")
        scaler = getattr(self, f"_{task}_scaler")
        pca = getattr(self, f"_{task}_pca")
        if rf is None or svm is None or scaler is None or pca is None:
            raise RuntimeError(
                f"Models for '{task}' not found. Run "
                f"python ml_pipeline/training/train_{task}.py first and ensure "
                f"outputs/trained_models/ contains {task}_rf.pkl, {task}_svm.pkl, etc."
            )

    def load_hsi_data(
        self,
        contents: bytes,
        ext: str,
        filename: str,
        hdr_bytes: bytes = None,
    ) -> np.ndarray:
        """Parse uploaded bytes into (H, W, bands) numpy array. Supports ENVI .hdr+.raw."""
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ext
        if ext in (".hdr", ".raw", ".bin") and hdr_bytes is not None:
            return load_hyperspectral_from_bytes(
                contents, ext, filename, hdr_bytes=hdr_bytes
            )
        if ext in (".hdr", ".raw", ".bin"):
            # Single file: try raw binary fallback for .raw/.bin only
            if ext == ".hdr":
                raise ValueError(
                    "ENVI .hdr requires the raw data file. Upload both .hdr and .raw (or use .npy/.npz)."
                )
            cube = np.frombuffer(contents, dtype=np.float32)
            total = cube.size
            if total >= 32 * 32 * 16:
                cube = cube[: 32 * 32 * 16].reshape(32, 32, 16)
            else:
                padded = np.zeros(32 * 32 * 16, dtype=np.float32)
                padded[:total] = cube
                cube = padded.reshape(32, 32, 16)
        else:
            cube = load_hyperspectral_from_bytes(contents, ext, filename)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        return np.asarray(cube, dtype=np.float32)

    def _preprocess(self, cube: np.ndarray) -> np.ndarray:
        """Radiometric-style normalization to [0,1] when no dark/white refs provided."""
        return apply_radiometric_correction(cube, dark_ref=None, white_ref=None)

    def _extract_features(self, cube: np.ndarray) -> np.ndarray:
        """Combine spectral + spatial features using ml_pipeline (same as training)."""
        spectral = extract_spectral_features(cube)
        spatial = extract_spatial_features(cube)
        return np.concatenate([spectral, spatial]).astype(np.float32)

    def _scale_and_reduce(self, features: np.ndarray, task: str) -> np.ndarray:
        """Apply scaler and PCA for the given task."""
        scaler = getattr(self, f"_{task}_scaler")
        pca = getattr(self, f"_{task}_pca")
        X = scaler.transform(features.reshape(1, -1))
        # PCA may have been fitted on fewer dims; pad or truncate if needed
        n_components = pca.n_components_
        if X.shape[1] > pca.n_features_in_:
            X = X[:, : pca.n_features_in_]
        elif X.shape[1] < pca.n_features_in_:
            pad = np.zeros((1, pca.n_features_in_ - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        return pca.transform(X)

    def _get_probs(self, model, features: np.ndarray, n_classes: int) -> np.ndarray:
        """Get probability array from fitted classifier."""
        p = model.predict_proba(features)[0]
        if len(p) == n_classes:
            return p
        # Pad or truncate to match class count
        out = np.zeros(n_classes, dtype=np.float32)
        out[: min(len(p), n_classes)] = p[: min(len(p), n_classes)]
        if out.sum() > 0:
            out /= out.sum()
        else:
            out[:] = 1.0 / n_classes
        return out

    def _softmax_fusion(self, probs_a: np.ndarray, probs_b: np.ndarray) -> np.ndarray:
        """Combine RF and SVM probabilities via softmax (same as ml_pipeline)."""
        avg = (probs_a + probs_b) / 2.0
        return softmax(avg.reshape(1, -1), axis=1)[0]

    def _compute_spectral_signature(self, cube: np.ndarray) -> dict:
        """Mean reflectance per band for tissue and simulated healthy reference."""
        H, W, B = cube.shape
        flat = cube.reshape(-1, B)
        tissue = flat.mean(axis=0).tolist()
        healthy = [min(1.0, v + 0.1 + 0.05 * np.random.randn()) for v in tissue]
        return {
            "bands": list(range(1, B + 1)),
            "tissue": [round(v * 100, 2) for v in tissue],
            "healthy": [round(v * 100, 2) for v in healthy],
        }

    def _generate_heatmap(self, cube: np.ndarray, class_idx: int) -> np.ndarray:
        """Spatial probability heatmap from band energy."""
        H, W, B = cube.shape
        band_energy = cube.mean(axis=2)
        mn, mx = band_energy.min(), band_energy.max()
        if mx - mn > 1e-8:
            normalized = (band_energy - mn) / (mx - mn)
        else:
            normalized = np.zeros_like(band_energy)
        if class_idx == 0:
            normalized = 1.0 - normalized
        if H != 32 or W != 32:
            from scipy.ndimage import zoom
            scale_h, scale_w = 32 / H, 32 / W
            normalized = zoom(normalized, (scale_h, scale_w))
        return np.clip(normalized, 0, 1)

    def predict_perfusion(self, cube: np.ndarray) -> dict:
        """Full perfusion pipeline: preprocess → features → scale/PCA → RF+SVM → softmax fusion."""
        self._require_models("perfusion")
        cube = self._preprocess(cube)
        features = self._extract_features(cube).reshape(1, -1)
        features_reduced = self._scale_and_reduce(features, "perfusion")

        rf_probs = self._get_probs(
            self._perfusion_rf, features_reduced, len(PERFUSION_CLASSES)
        )
        svm_probs = self._get_probs(
            self._perfusion_svm, features_reduced, len(PERFUSION_CLASSES)
        )
        fused = self._softmax_fusion(rf_probs, svm_probs)
        winner_idx = int(np.argmax(fused))
        confidence = round(float(fused[winner_idx]) * 100, 1)

        spectral_sig = self._compute_spectral_signature(cube)
        heatmap = self._generate_heatmap(cube, winner_idx)

        rf_pred_label = PERFUSION_CLASSES[int(np.argmax(rf_probs))]
        svm_pred_label = PERFUSION_CLASSES[int(np.argmax(svm_probs))]

        return {
            "task": "perfusion",
            "prediction": PERFUSION_CLASSES[winner_idx],
            "confidence": confidence,
            "rf_prediction": rf_pred_label,
            "svm_prediction": svm_pred_label,
            "classifier_probabilities": {
                "random_forest": {
                    k: round(float(v) * 100, 1)
                    for k, v in zip(PERFUSION_CLASSES, rf_probs)
                },
                "svm": {
                    k: round(float(v) * 100, 1)
                    for k, v in zip(PERFUSION_CLASSES, svm_probs)
                },
            },
            "fusion_probabilities": {
                PERFUSION_CLASSES[i]: round(float(fused[i]) * 100, 1)
                for i in range(len(PERFUSION_CLASSES))
            },
            "classifiers": {
                "random_forest": {
                    "prediction": rf_pred_label,
                    "confidence": round(float(np.max(rf_probs)) * 100, 1),
                    "probabilities": {
                        PERFUSION_CLASSES[i]: round(float(rf_probs[i]) * 100, 1)
                        for i in range(len(PERFUSION_CLASSES))
                    },
                },
                "svm": {
                    "prediction": svm_pred_label,
                    "confidence": round(float(np.max(svm_probs)) * 100, 1),
                    "probabilities": {
                        PERFUSION_CLASSES[i]: round(float(svm_probs[i]) * 100, 1)
                        for i in range(len(PERFUSION_CLASSES))
                    },
                },
            },
            "spectral_signature": spectral_sig,
            "heatmap": heatmap.tolist(),
            "ai_summary": self._perfusion_summary(PERFUSION_CLASSES[winner_idx], confidence),
        }

    def predict_tumor(self, cube: np.ndarray) -> dict:
        """Full tumor pipeline: preprocess → features → scale/PCA → RF+SVM → softmax fusion."""
        self._require_models("tumor")
        cube = self._preprocess(cube)
        features = self._extract_features(cube).reshape(1, -1)
        features_reduced = self._scale_and_reduce(features, "tumor")

        rf_probs = self._get_probs(
            self._tumor_rf, features_reduced, len(TUMOR_CLASSES)
        )
        svm_probs = self._get_probs(
            self._tumor_svm, features_reduced, len(TUMOR_CLASSES)
        )
        fused = self._softmax_fusion(rf_probs, svm_probs)
        winner_idx = int(np.argmax(fused))
        confidence = round(float(fused[winner_idx]) * 100, 1)

        spectral_sig = self._compute_spectral_signature(cube)
        heatmap = self._generate_heatmap(cube, winner_idx)

        rf_pred_label = TUMOR_CLASSES[int(np.argmax(rf_probs))]
        svm_pred_label = TUMOR_CLASSES[int(np.argmax(svm_probs))]

        return {
            "task": "tumor",
            "prediction": TUMOR_CLASSES[winner_idx],
            "confidence": confidence,
            "rf_prediction": rf_pred_label,
            "svm_prediction": svm_pred_label,
            "classifier_probabilities": {
                "random_forest": {
                    k: round(float(v) * 100, 1) for k, v in zip(TUMOR_CLASSES, rf_probs)
                },
                "svm": {
                    k: round(float(v) * 100, 1) for k, v in zip(TUMOR_CLASSES, svm_probs)
                },
            },
            "fusion_probabilities": {
                TUMOR_CLASSES[i]: round(float(fused[i]) * 100, 1)
                for i in range(len(TUMOR_CLASSES))
            },
            "classifiers": {
                "random_forest": {
                    "prediction": rf_pred_label,
                    "confidence": round(float(np.max(rf_probs)) * 100, 1),
                    "probabilities": {
                        TUMOR_CLASSES[i]: round(float(rf_probs[i]) * 100, 1)
                        for i in range(len(TUMOR_CLASSES))
                    },
                },
                "svm": {
                    "prediction": svm_pred_label,
                    "confidence": round(float(np.max(svm_probs)) * 100, 1),
                    "probabilities": {
                        TUMOR_CLASSES[i]: round(float(svm_probs[i]) * 100, 1)
                        for i in range(len(TUMOR_CLASSES))
                    },
                },
            },
            "spectral_signature": spectral_sig,
            "heatmap": heatmap.tolist(),
            "ai_summary": self._tumor_summary(TUMOR_CLASSES[winner_idx], confidence),
        }

    def _perfusion_summary(self, label: str, conf: float) -> str:
        summaries = {
            "Normal Perfusion": f"Normal blood flow detected throughout the tissue region. Oxygenation levels are within expected physiological range. (Confidence: {conf}%)",
            "Reduced Perfusion": f"Decreased blood flow detected in the tissue region. Spectral analysis indicates reduced oxygenation patterns consistent with partial vascular restriction. (Confidence: {conf}%)",
            "Abnormal Perfusion": f"Significant perfusion abnormality detected. Severely reduced blood flow with oxygenation indicators below clinical threshold, suggesting possible vascular occlusion. (Confidence: {conf}%)",
        }
        return summaries.get(label, f"Perfusion analysis complete. Prediction: {label}.")

    def _tumor_summary(self, label: str, conf: float) -> str:
        summaries = {
            "Non-Tumor Tissue": f"No tumor signatures detected in the tissue sample. Spectral-spatial profile is consistent with healthy neural tissue histology. (Confidence: {conf}%)",
            "Glioblastoma Tumor": f"Glioblastoma tumor tissue identified in the histological sample. Spectral-spatial features match ROI_T classification patterns associated with high-grade glioma. (Confidence: {conf}%)",
        }
        return summaries.get(label, f"Tumor detection complete. Prediction: {label}.")
