
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np


def test_dummy_prediction():
    cube = np.random.rand(16, 16, 8)
    assert cube.shape[0] == 16


def test_config_imports():
    from config.config import (
        PATCH_SIZE, STRIDE, TRAINED_MODELS_DIR,
        DATASETS_DIR, MODEL_OUTPUT, DATASET_PATH,
    )
    assert PATCH_SIZE == 32
    assert STRIDE == 16
    assert MODEL_OUTPUT == TRAINED_MODELS_DIR
    assert DATASET_PATH == DATASETS_DIR


def test_preprocessing_imports():
    from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction
    from ml_pipeline.preprocessing.patch_extraction import extract_patches, select_informative_bands
    cube = np.random.rand(64, 64, 20).astype("float32")
    corrected = apply_radiometric_correction(cube)
    assert corrected.min() >= 0.0 and corrected.max() <= 1.0
    selected = select_informative_bands(corrected, n_bands=10)
    assert selected.shape[2] == 10


def test_feature_extraction():
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
    patch = np.random.rand(32, 32, 16).astype("float32")
    spec = extract_spectral_features(patch)
    spat = extract_spatial_features(patch)
    assert spec.ndim == 1 and spec.size > 0
    assert spat.ndim == 1 and spat.size > 0


def test_softmax_fusion():
    from ml_pipeline.fusion.softmax_fusion import softmax
    x = np.array([[1.0, 2.0, 3.0]])
    out = softmax(x, axis=1)
    assert abs(out.sum() - 1.0) < 1e-5


def test_patch_extraction_yields_3d():
    from ml_pipeline.preprocessing.patch_extraction import extract_patches
    cube = np.random.rand(64, 64, 16).astype("float32")
    for r, c, patch in extract_patches(cube, patch_size=32, stride=32):
        assert patch.ndim == 3
        assert patch.shape == (32, 32, 16)


def test_audit_logger_import():
    from backend.audit.audit_logger import AuditLogger
    assert AuditLogger is not None


def test_monitoring_import():
    from backend.monitoring.metrics import record_prediction, record_latency
    assert callable(record_prediction)


def test_hsi_loader_import():
    from ml_pipeline.data_loader.hsi_loader import (
        _warn_large_cube, _validate_cube_content,
    )
    cube = np.random.rand(32, 32, 16).astype("float32")
    _warn_large_cube(cube)   # should not raise


def test_run_pipeline_on_cube_missing_or_present_models():
    """
    If trained models are present → full inference must succeed with correct schema.
    If models are missing → FileNotFoundError must be raised (not a crash).
    """
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    from config.config import TRAINED_MODELS_DIR
    import numpy as np

    cube = np.random.rand(64, 64, 20).astype("float32")
    models_present = all(
        (TRAINED_MODELS_DIR / f"tumor_{m}.pkl").exists()
        for m in ("rf", "svm", "scaler", "pca")
    )

    if models_present:
        result = run_pipeline_on_cube(cube, "tumor")
        assert result["task"] == "tumor"
        assert result["prediction"] in ["Non-Tumor Tissue", "Glioblastoma Tumor"]
        assert 0.0 <= result["confidence"] <= 100.0
        assert "heatmap" in result
    else:
        try:
            run_pipeline_on_cube(cube, "tumor")
            assert False, "Expected FileNotFoundError when models are missing"
        except FileNotFoundError:
            pass  # correct behaviour

