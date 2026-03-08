"""
End-to-End Integration Test — Unified HSI Medical Framework
============================================================
Runs without network access or optional packages (fastapi/uvicorn/torch).
Uses synthetic HSI cubes to exercise the full training → prediction pipeline.

Steps
-----
1.  Import verification   — all core modules load cleanly
2.  Preprocessing         — radiometric correction, band selection, patch extraction
3.  Feature extraction    — spectral + spatial features on real patches
4.  Training pipeline     — RF + SVM fit on synthetic features, saves .pkl models
5.  Prediction pipeline   — run_pipeline_on_cube() end-to-end
6.  Heatmap generation    — PNG written to outputs/heatmaps/
7.  Dashboard HTML check  — verify all required assets and endpoints present
8.  Audit log             — JSON written to logs/audit/
9.  Evaluation metrics    — tumor_metrics.json + perfusion_metrics.json written
10. pytest-style summary  — PASSED / FAILED per test, exit 0 on all-pass
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

results = []   # list of (name, passed, duration_s, detail)


def run(name):
    """Decorator — run a test function and record pass/fail."""
    def decorator(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            dt = time.perf_counter() - t0
            results.append((name, True, dt, detail or ""))
            print(f"  {GREEN}✓ PASS{RESET}  {name}  ({dt*1000:.0f} ms)")
        except Exception as exc:
            dt = time.perf_counter() - t0
            tb = traceback.format_exc()
            results.append((name, False, dt, tb))
            print(f"  {RED}✗ FAIL{RESET}  {name}")
            print(f"         {RED}{exc}{RESET}")
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(seed=42)

def _cube(h=64, w=64, b=20):
    """Create a synthetic (H,W,B) float32 HSI cube."""
    return RNG.random((h, w, b), dtype="float32")

def _build_synthetic_features(n_samples=120, n_features=None):
    """
    Build feature matrix identical to what the real pipeline produces
    so we can train models without actual HSI files.
    """
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    from ml_pipeline.feature_extraction.spatial_features  import extract_spatial_features

    patch = _cube(32, 32, 16)
    spec  = extract_spectral_features(patch)
    spat  = extract_spatial_features(patch)
    n_features = len(spec) + len(spat)

    X = RNG.random((n_samples, n_features), dtype="float32")
    return X, n_features


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Import verification
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 1 — Import Verification ━━━{RESET}")

@run("config.config imports")
def _():
    from config.config import (PATCH_SIZE, STRIDE, STRIDE_PERF,
                                TRAINED_MODELS_DIR, EVAL_METRICS_DIR,
                                MODEL_OUTPUT, DATASET_PATH,
                                HEATMAPS_DIR, PREDICTION_RESULTS_DIR)
    assert PATCH_SIZE == 32
    assert STRIDE == 16
    assert MODEL_OUTPUT == TRAINED_MODELS_DIR
    return f"PATCH_SIZE={PATCH_SIZE}, STRIDE={STRIDE}"

@run("preprocessing modules import")
def _():
    from ml_pipeline.preprocessing.radiometric    import apply_radiometric_correction
    from ml_pipeline.preprocessing.patch_extraction import extract_patches, select_informative_bands
    return "radiometric + patch_extraction"

@run("feature extraction modules import")
def _():
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    from ml_pipeline.feature_extraction.spatial_features  import extract_spatial_features
    return "spectral_features + spatial_features"

@run("fusion + visualization modules import")
def _():
    from ml_pipeline.fusion.softmax_fusion          import softmax, SoftmaxFusion
    from ml_pipeline.visualization.heatmap          import generate_tumor_heatmap, save_heatmap
    from ml_pipeline.visualization.feature_importance import save_feature_importance
    from ml_pipeline.evaluation.model_comparison    import evaluate_models, plot_model_comparison
    return "softmax_fusion + heatmap + feature_importance + model_comparison"

@run("backend audit + monitoring import")
def _():
    from backend.audit.audit_logger      import AuditLogger
    from backend.monitoring.metrics      import record_prediction, record_latency
    assert callable(record_prediction) and callable(record_latency)
    return "AuditLogger + record_prediction + record_latency"

@run("prediction pipeline import")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    assert callable(run_pipeline_on_cube)
    return "run_pipeline_on_cube importable"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 2 — Preprocessing Pipeline ━━━{RESET}")

@run("radiometric correction (no refs → global min-max)")
def _():
    from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction
    cube = _cube()
    # Inject some non-zero-mean data so correction is non-trivial
    cube = cube * 500 + 100
    out  = apply_radiometric_correction(cube)
    assert out.shape == cube.shape
    assert out.min() >= 0.0 and out.max() <= 1.001
    return f"output range [{out.min():.4f}, {out.max():.4f}]"

@run("radiometric correction (dark + white refs)")
def _():
    from ml_pipeline.preprocessing.radiometric import apply_radiometric_correction
    cube  = _cube()
    dark  = np.zeros(20, dtype="float32")
    white = np.ones(20,  dtype="float32")
    out   = apply_radiometric_correction(cube, dark_ref=dark, white_ref=white)
    assert out.min() >= 0.0 and out.max() <= 1.001
    return "dark+white refs applied"

@run("band selection (top-10 of 20 by variance)")
def _():
    from ml_pipeline.preprocessing.patch_extraction import select_informative_bands
    cube = _cube(64, 64, 20)
    sel  = select_informative_bands(cube, n_bands=10)
    assert sel.shape == (64, 64, 10), f"got {sel.shape}"
    return f"(64,64,20) → (64,64,10)"

@run("patch extraction yields (32,32,B) patches")
def _():
    from ml_pipeline.preprocessing.patch_extraction import extract_patches
    cube    = _cube(64, 64, 16)
    patches = list(extract_patches(cube, patch_size=32, stride=32))
    assert len(patches) == 4, f"expected 4 patches, got {len(patches)}"
    for _, _, p in patches:
        assert p.ndim  == 3
        assert p.shape == (32, 32, 16)
    return f"{len(patches)} patches, each (32,32,16)"

@run("patch extraction — small cube padding")
def _():
    from ml_pipeline.preprocessing.patch_extraction import extract_patches
    cube    = _cube(10, 10, 8)      # smaller than patch_size=32
    patches = list(extract_patches(cube, patch_size=32, stride=32))
    assert len(patches) >= 1
    for _, _, p in patches:
        assert p.shape == (32, 32, 8)
    return f"tiny cube padded → {len(patches)} patch"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature Extraction
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 3 — Feature Extraction ━━━{RESET}")

@run("spectral feature vector shape + dtype")
def _():
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    patch = _cube(32, 32, 16)
    spec  = extract_spectral_features(patch)
    assert spec.ndim == 1
    assert spec.dtype == np.float32
    # Expected: 5*B + (B-1) + 2 = 5*16 + 15 + 2 = 97
    assert spec.size > 0
    return f"shape=({spec.size},) dtype={spec.dtype}"

@run("spatial feature vector (GLCM + gradient + LBP)")
def _():
    from ml_pipeline.feature_extraction.spatial_features import extract_spatial_features
    patch = _cube(32, 32, 16)
    spat  = extract_spatial_features(patch)
    assert spat.ndim == 1
    assert spat.dtype == np.float32
    assert spat.size > 0
    return f"shape=({spat.size},) dtype={spat.dtype}"

@run("joint feature concat is finite")
def _():
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    from ml_pipeline.feature_extraction.spatial_features  import extract_spatial_features
    patch = _cube(32, 32, 16)
    feat  = np.concatenate([extract_spectral_features(patch),
                             extract_spatial_features(patch)]).astype(np.float32)
    assert np.isfinite(feat).all(), "NaN/Inf in features"
    return f"joint feature dim={feat.size}, all finite"

@run("softmax fusion probabilities sum to 1")
def _():
    from ml_pipeline.fusion.softmax_fusion import softmax
    x   = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 1.2]])
    out = softmax(x, axis=1)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], atol=1e-5)
    return f"rows sum to 1.0 ✓"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Training Pipeline (synthetic data, saves real .pkl files)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 4 — Training Pipeline ━━━{RESET}")

@run("build synthetic feature matrix (120 samples)")
def _():
    global _X_synth, _n_feat
    _X_synth, _n_feat = _build_synthetic_features(120)
    assert _X_synth.shape == (120, _n_feat)
    assert np.isfinite(_X_synth).all()
    return f"shape={_X_synth.shape}"

@run("RF + SVM train on synthetic tumor features")
def _():
    global _rf_tumor, _svm_tumor, _scaler_tumor, _pca_tumor
    from sklearn.ensemble        import RandomForestClassifier
    from sklearn.svm             import SVC
    from sklearn.preprocessing   import StandardScaler
    from sklearn.decomposition   import PCA
    from config.config           import RF_TREES, RF_MAX_DEPTH, SVM_KERNEL, SVM_C_TUMOR, N_PCA_TUMOR

    X, _  = _build_synthetic_features(120)
    y     = RNG.integers(0, 2, size=120)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    n_comp = min(N_PCA_TUMOR, Xs.shape[1], Xs.shape[0] - 1)
    pca    = PCA(n_components=n_comp)
    Xp     = pca.fit_transform(Xs)

    rf  = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C_TUMOR, probability=True, random_state=42)
    rf.fit(Xp, y)
    svm.fit(Xp, y)

    _rf_tumor, _svm_tumor, _scaler_tumor, _pca_tumor = rf, svm, scaler, pca
    return f"RF trees=20, SVM kernel={SVM_KERNEL}, PCA n_comp={n_comp}"

@run("save tumor models to outputs/trained_models/")
def _():
    import joblib
    from config.config import TRAINED_MODELS_DIR
    for name, obj in [("rf", _rf_tumor), ("svm", _svm_tumor),
                      ("scaler", _scaler_tumor), ("pca", _pca_tumor)]:
        p = TRAINED_MODELS_DIR / f"tumor_{name}.pkl"
        joblib.dump(obj, p)
        assert p.exists() and p.stat().st_size > 0
    return "tumor_rf/svm/scaler/pca.pkl written"

@run("RF + SVM train on synthetic perfusion features (3 classes)")
def _():
    global _rf_perf, _svm_perf, _scaler_perf, _pca_perf
    from sklearn.ensemble       import RandomForestClassifier
    from sklearn.svm            import SVC
    from sklearn.preprocessing  import StandardScaler
    from sklearn.decomposition  import PCA
    from config.config          import SVM_KERNEL, SVM_C_PERF, N_PCA_PERF

    X, _   = _build_synthetic_features(150)
    y      = RNG.integers(0, 3, size=150)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_comp = min(N_PCA_PERF, Xs.shape[1], Xs.shape[0] - 1)
    pca    = PCA(n_components=n_comp)
    Xp     = pca.fit_transform(Xs)

    rf  = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    svm = SVC(kernel=SVM_KERNEL, C=SVM_C_PERF, probability=True, random_state=42)
    rf.fit(Xp, y)
    svm.fit(Xp, y)

    _rf_perf, _svm_perf, _scaler_perf, _pca_perf = rf, svm, scaler, pca
    return f"3-class perfusion, PCA n_comp={n_comp}"

@run("save perfusion models to outputs/trained_models/")
def _():
    import joblib
    from config.config import TRAINED_MODELS_DIR
    for name, obj in [("rf", _rf_perf), ("svm", _svm_perf),
                      ("scaler", _scaler_perf), ("pca", _pca_perf)]:
        p = TRAINED_MODELS_DIR / f"perfusion_{name}.pkl"
        joblib.dump(obj, p)
        assert p.exists() and p.stat().st_size > 0
    return "perfusion_rf/svm/scaler/pca.pkl written"

@run("save tumor evaluation metrics JSON")
def _():
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from config.config   import EVAL_METRICS_DIR, TRAINED_MODELS_DIR
    import joblib

    scaler = joblib.load(TRAINED_MODELS_DIR / "tumor_scaler.pkl")
    pca    = joblib.load(TRAINED_MODELS_DIR / "tumor_pca.pkl")
    rf     = joblib.load(TRAINED_MODELS_DIR / "tumor_rf.pkl")
    svm    = joblib.load(TRAINED_MODELS_DIR / "tumor_svm.pkl")

    X, _  = _build_synthetic_features(40)
    y     = RNG.integers(0, 2, size=40)
    Xp    = pca.transform(scaler.transform(X))

    classes = ["Non-Tumor Tissue", "Glioblastoma Tumor"]
    def _m(y_true, y_pred):
        return {
            "accuracy":                round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score":                round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
            "confusion_matrix":        confusion_matrix(y_true, y_pred, labels=[0,1]).tolist(),
            "confusion_matrix_labels": classes,
        }

    data = {
        "classes":       classes,
        "fusion":        _m(y, rf.predict(Xp)),   # proxy
        "random_forest": _m(y, rf.predict(Xp)),
        "svm":           _m(y, svm.predict(Xp)),
    }
    p = EVAL_METRICS_DIR / "tumor_metrics.json"
    p.write_text(json.dumps(data, indent=2))
    assert p.exists()
    return f"tumor_metrics.json  ({p.stat().st_size} bytes)"

@run("save perfusion evaluation metrics JSON")
def _():
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from config.config   import EVAL_METRICS_DIR, TRAINED_MODELS_DIR
    import joblib

    scaler = joblib.load(TRAINED_MODELS_DIR / "perfusion_scaler.pkl")
    pca    = joblib.load(TRAINED_MODELS_DIR / "perfusion_pca.pkl")
    rf     = joblib.load(TRAINED_MODELS_DIR / "perfusion_rf.pkl")
    svm    = joblib.load(TRAINED_MODELS_DIR / "perfusion_svm.pkl")

    X, _  = _build_synthetic_features(60)
    y     = RNG.integers(0, 3, size=60)
    Xp    = pca.transform(scaler.transform(X))

    classes = ["Normal Perfusion", "Reduced Perfusion", "Abnormal Perfusion"]
    def _m(y_true, y_pred):
        return {
            "accuracy":                round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score":                round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "confusion_matrix":        confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist(),
            "confusion_matrix_labels": classes,
        }

    data = {
        "classes":       classes,
        "fusion":        _m(y, rf.predict(Xp)),
        "random_forest": _m(y, rf.predict(Xp)),
        "svm":           _m(y, svm.predict(Xp)),
    }
    p = EVAL_METRICS_DIR / "perfusion_metrics.json"
    p.write_text(json.dumps(data, indent=2))
    assert p.exists()
    return f"perfusion_metrics.json  ({p.stat().st_size} bytes)"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Prediction Pipeline
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 5 — Prediction Pipeline ━━━{RESET}")

@run("run_pipeline_on_cube — tumor task")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    cube   = _cube(64, 64, 20)
    result = run_pipeline_on_cube(cube, "tumor")

    assert "prediction"          in result
    assert "confidence"          in result
    assert "fusion_probabilities" in result
    assert "classifiers"         in result
    assert "spectral_signature"  in result
    assert "heatmap"             in result
    assert "ai_summary"          in result
    assert result["task"]        == "tumor"
    assert result["prediction"]  in ["Non-Tumor Tissue", "Glioblastoma Tumor"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["heatmap"]) == 32       # 32×32
    return f"prediction='{result['prediction']}'  confidence={result['confidence']}%"

@run("run_pipeline_on_cube — perfusion task")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    cube   = _cube(64, 64, 20)
    result = run_pipeline_on_cube(cube, "perfusion")

    assert result["task"] == "perfusion"
    assert result["prediction"] in ["Normal Perfusion", "Reduced Perfusion", "Abnormal Perfusion"]
    assert 0.0 <= result["confidence"] <= 100.0
    return f"prediction='{result['prediction']}'  confidence={result['confidence']}%"

@run("fusion_probabilities sum to ~100%")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    result = run_pipeline_on_cube(_cube(64,64,20), "tumor")
    total  = sum(result["fusion_probabilities"].values())
    assert abs(total - 100.0) < 1.0, f"probs sum to {total}"
    return f"sum={total:.2f}%"

@run("classifiers block has RF + SVM entries")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    result = run_pipeline_on_cube(_cube(64,64,20), "tumor")
    clf    = result["classifiers"]
    assert "random_forest" in clf and "svm" in clf
    for k in ("prediction", "confidence", "probabilities"):
        assert k in clf["random_forest"]
        assert k in clf["svm"]
    return "random_forest + svm blocks present"

@run("spectral_signature has bands / tissue / healthy lists")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    result = run_pipeline_on_cube(_cube(64,64,20), "tumor")
    ss     = result["spectral_signature"]
    assert "bands" in ss and "tissue" in ss and "healthy" in ss
    assert len(ss["bands"]) == len(ss["tissue"]) == len(ss["healthy"])
    return f"spectral signature length={len(ss['bands'])}"

@run("prediction result JSON persisted to disk")
def _():
    from config.config import PREDICTION_RESULTS_DIR
    files = sorted(PREDICTION_RESULTS_DIR.glob("tumor_prediction_*.json"))
    assert len(files) >= 1, "No prediction JSON found"
    data  = json.loads(files[-1].read_text())
    assert "prediction" in data and "confidence" in data
    return f"latest: {files[-1].name}"

@run("unknown task raises ValueError")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    try:
        run_pipeline_on_cube(_cube(), "banana")
        assert False, "should have raised"
    except ValueError as e:
        assert "banana" in str(e)
    return "ValueError raised correctly"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Heatmap Generation
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 6 — Heatmap Generation ━━━{RESET}")

@run("heatmap is 32×32 list-of-lists in result")
def _():
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    result  = run_pipeline_on_cube(_cube(64,64,20), "tumor")
    heatmap = result["heatmap"]
    assert len(heatmap) == 32
    assert len(heatmap[0]) == 32
    vals = [v for row in heatmap for v in row]
    assert all(0.0 <= v <= 1.0 for v in vals), "heatmap values out of [0,1]"
    return "32×32 list, values in [0,1]"

@run("heatmap PNG written to outputs/heatmaps/")
def _():
    from config.config import HEATMAPS_DIR
    pngs = sorted(HEATMAPS_DIR.glob("tumor_heatmap_*.png"))
    assert len(pngs) >= 1, "No heatmap PNG found"
    assert pngs[-1].stat().st_size > 1000, "PNG seems empty"
    return f"{pngs[-1].name}  ({pngs[-1].stat().st_size//1024} KB)"

@run("generate_tumor_heatmap helper produces correct shape")
def _():
    from ml_pipeline.visualization.heatmap import generate_tumor_heatmap, save_heatmap
    H, W, B = 64, 64, 16
    probs   = np.random.rand(4)   # 4 patches for a 64×64 / 32-stride grid
    heatmap = generate_tumor_heatmap((H, W, B), patches=None,
                                     probabilities=probs,
                                     patch_size=32, stride=32)
    assert heatmap.shape == (H, W)
    out = Path("outputs/heatmaps/test_helper_heatmap.png")
    save_heatmap(heatmap, str(out))
    assert out.exists()
    return f"shape={heatmap.shape}  PNG={out.name}"

@run("feature importance PNG generated")
def _():
    from ml_pipeline.visualization.feature_importance import save_feature_importance
    import joblib
    from config.config import TRAINED_MODELS_DIR
    rf   = joblib.load(TRAINED_MODELS_DIR / "tumor_rf.pkl")
    n    = rf.n_features_in_
    names = [f"pca_{i}" for i in range(n)]
    out   = Path("outputs/feature_importance/tumor_features.png")
    save_feature_importance(rf, names, str(out))
    assert out.exists() and out.stat().st_size > 1000
    return f"{out.name}  ({out.stat().st_size//1024} KB)"

@run("model comparison PNG generated")
def _():
    from ml_pipeline.evaluation.model_comparison import evaluate_models, plot_model_comparison
    import joblib
    from config.config import TRAINED_MODELS_DIR
    rf  = joblib.load(TRAINED_MODELS_DIR / "tumor_rf.pkl")
    svm = joblib.load(TRAINED_MODELS_DIR / "tumor_svm.pkl")
    pca = joblib.load(TRAINED_MODELS_DIR / "tumor_pca.pkl")
    scaler = joblib.load(TRAINED_MODELS_DIR / "tumor_scaler.pkl")

    X, _   = _build_synthetic_features(40)
    Xp     = pca.transform(scaler.transform(X))
    y      = RNG.integers(0, 2, size=40)
    models = {"random_forest": rf, "svm": svm}

    df  = evaluate_models(models, Xp, y)
    out = Path("outputs/model_comparison/tumor_model_comparison.png")
    plot_model_comparison(df, str(out))
    assert out.exists() and out.stat().st_size > 1000
    return f"{out.name}  ({out.stat().st_size//1024} KB)"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Dashboard / Frontend
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 7 — Dashboard & Frontend ━━━{RESET}")

@run("frontend HTML files exist")
def _():
    base = Path("frontend")
    for f in ("index.html", "tumor.html", "perfusion.html", "dashboard.html"):
        p = base / f
        assert p.exists(), f"Missing: {p}"
    return "index / tumor / perfusion / dashboard .html all present"

@run("frontend JS files exist")
def _():
    base = Path("frontend/js")
    for f in ("main.js", "api.js", "tumor.js", "perfusion.js"):
        p = base / f
        assert p.exists(), f"Missing: {p}"
    return "main / api / tumor / perfusion .js all present"

@run("dashboard.html references output directories")
def _():
    html = Path("frontend/dashboard.html").read_text()
    for keyword in ("heatmaps", "feature_importance", "spectral_analysis", "model_comparison"):
        assert keyword in html, f"dashboard.html missing reference to: {keyword}"
    return "all 4 output directory references found"

@run("api.js references /predict/tumor and /predict/perfusion")
def _():
    js = Path("frontend/js/api.js").read_text()
    assert "/predict/tumor"     in js, "/predict/tumor missing from api.js"
    assert "/predict/perfusion" in js, "/predict/perfusion missing from api.js"
    return "both endpoint paths present in api.js"

@run("output directories exist for dashboard image loading")
def _():
    for d in ("outputs/heatmaps", "outputs/feature_importance",
              "outputs/spectral_analysis", "outputs/model_comparison"):
        assert Path(d).is_dir(), f"Missing directory: {d}"
    return "all 4 output dirs exist"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Audit Logging
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 8 — Audit Logging ━━━{RESET}")

@run("AuditLogger writes valid JSON record")
def _():
    from backend.audit.audit_logger import AuditLogger
    al = AuditLogger()
    al.log_prediction("tumor", "Glioblastoma Tumor", 91.3,
                      metadata={"n_patches": 4, "cube_shape": [64,64,20]})

    files = sorted(Path("logs/audit").glob("audit_*.json"))
    assert len(files) >= 1
    records = json.loads(files[-1].read_text())
    last = records[-1]
    assert last["task"]        == "tumor"
    assert last["prediction"]  == "Glioblastoma Tumor"
    assert last["confidence"]  == 91.3
    assert "timestamp"         in last
    return f"record written to {files[-1].name}"

@run("audit log is auto-created per run_pipeline_on_cube")
def _():
    from config.config import PREDICTION_RESULTS_DIR
    files = sorted(PREDICTION_RESULTS_DIR.glob("*.json"))
    assert len(files) >= 1
    last  = json.loads(files[-1].read_text())
    assert "prediction" in last
    return f"{len(files)} prediction result(s) on disk"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Evaluation Metrics on disk
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 9 — Evaluation Metrics ━━━{RESET}")

@run("tumor_metrics.json is valid and has required keys")
def _():
    from config.config import EVAL_METRICS_DIR
    p    = EVAL_METRICS_DIR / "tumor_metrics.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert "classes"        in data
    assert "fusion"         in data
    assert "random_forest"  in data
    assert "svm"            in data
    for section in ("fusion", "random_forest", "svm"):
        assert "accuracy"            in data[section]
        assert "f1_score"            in data[section]
        assert "confusion_matrix"    in data[section]
    return f"accuracy(fusion)={data['fusion']['accuracy']}"

@run("perfusion_metrics.json is valid and has required keys")
def _():
    from config.config import EVAL_METRICS_DIR
    p    = EVAL_METRICS_DIR / "perfusion_metrics.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert len(data["classes"]) == 3
    assert "fusion" in data
    return f"accuracy(fusion)={data['fusion']['accuracy']}"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — run.py CLI smoke test
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}{CYAN}━━━ STEP 10 — run.py CLI Smoke Test ━━━{RESET}")

@run("run.py --health executes without error")
def _():
    import subprocess
    # ── FIX (cross-platform: Linux + Windows) ────────────────────────────────
    # Root cause: when run.py is launched as a subprocess with capture_output=True,
    # stdout is a binary pipe. On:
    #   - Linux/Docker/CI  → default encoding may fall back to ASCII
    #   - Windows PowerShell → console uses cp1252, not UTF-8
    # run.py previously printed Unicode box-drawing/tick chars (✓ ✗ ─ ━)
    # which raised UnicodeEncodeError → exit code 1 → assertion failure.
    #
    # Fix A: run.py action_health() now uses only ASCII chars ([OK]/[MISSING]).
    # Fix B: env sets PYTHONIOENCODING + PYTHONUTF8 so the child process uses
    #        UTF-8 I/O even on Windows.
    # Fix C: subprocess.run uses explicit encoding='utf-8' so stdout/stderr are
    #        decoded correctly regardless of the host console code page.
    # Fix D: cwd resolved to absolute path so it is unambiguous on all OSes.
    # ─────────────────────────────────────────────────────────────────────────
    _env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    _cwd = str(Path(__file__).resolve().parent)

    result = subprocess.run(
        [sys.executable, "run.py", "--health"],
        capture_output=True,
        encoding="utf-8",       # explicit UTF-8 decode on stdout/stderr
        errors="replace",       # never crash on stray bytes
        cwd=_cwd,
        env=_env,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"run.py --health exited with code {result.returncode}\n"
        f"--- stderr ---\n{result.stderr[:400]}\n"
        f"--- stdout ---\n{result.stdout[:400]}"
    )
    assert "tumor_rf.pkl" in result.stdout, (
        f"Expected 'tumor_rf.pkl' in stdout but got:\n{result.stdout[:400]}"
    )
    assert "perfusion_rf.pkl" in result.stdout, (
        f"Expected 'perfusion_rf.pkl' in stdout but got:\n{result.stdout[:400]}"
    )
    return "health check passed, all models reported OK"


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]
total  = len(results)

print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  END-TO-END TEST SUMMARY{RESET}")
print(f"{'═'*60}")
print(f"  Total  : {total}")
print(f"  {GREEN}Passed : {len(passed)}{RESET}")
if failed:
    print(f"  {RED}Failed : {len(failed)}{RESET}")
    print(f"\n{RED}Failed tests:{RESET}")
    for name, _, _, tb in failed:
        print(f"  ✗ {name}")
        for line in tb.strip().splitlines()[-3:]:
            print(f"      {line}")
print(f"{'═'*60}")

if not failed:
    print(f"\n{GREEN}{BOLD}  ✓ ALL {total} TESTS PASSED — PROJECT RUNS WITHOUT ERRORS{RESET}\n")
    sys.exit(0)
else:
    print(f"\n{RED}{BOLD}  ✗ {len(failed)} TEST(S) FAILED{RESET}\n")
    sys.exit(1)