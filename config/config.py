"""
Central Configuration — Unified HSI Medical Framework
All tunable parameters live here. Import with:
    from config.config import CFG
or directly:
    from config.config import PATCH_SIZE, STRIDE, ...
"""

from pathlib import Path

# ─── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ─── Patch extraction ──────────────────────────────────────────────────────────
PATCH_SIZE   = 32          # Spatial patch size (H = W = 32)
STRIDE       = 16          # Grid stride for tumor task
STRIDE_PERF  = 32          # Grid stride for perfusion task
MAX_PATCHES  = 500         # Hard cap on patches per cube/ROI at training time

# ─── Spectral band selection ───────────────────────────────────────────────────
N_INFORMATIVE_BANDS = 16   # 0 = use all bands; > 0 = keep top-N by variance

# ─── Random Forest ─────────────────────────────────────────────────────────────
RF_TREES      = 300
RF_MAX_DEPTH  = 20
RF_JOBS       = -1

# ─── SVM ───────────────────────────────────────────────────────────────────────
SVM_KERNEL    = "rbf"
SVM_C_TUMOR   = 50.0
SVM_C_PERF    = 10.0

# ─── PCA ───────────────────────────────────────────────────────────────────────
N_PCA_TUMOR   = 25
N_PCA_PERF    = 40

# ─── Majority voting ───────────────────────────────────────────────────────────
# If True, final class label is determined by majority vote across patch
# predictions; probabilities are still averaged for the frontend display.
USE_MAJORITY_VOTE = True

# ─── Hyperspectral cube validation ────────────────────────────────────────────
MIN_BANDS = 10              # minimum number of spectral bands; raise if below this

# ─── Large-cube warning threshold ─────────────────────────────────────────────
LARGE_CUBE_ELEMENTS = int(2e9)   # warn if cube.size exceeds this

# ─── Batch patch inference ─────────────────────────────────────────────────────
# Number of patches to process at once during predict_proba calls.
# Reduces peak RAM on large cubes.  Set to 0 or None to disable batching.
PATCH_BATCH_SIZE = 64

# ─── Allowed upload extensions ─────────────────────────────────────────────────
ALLOWED_UPLOAD_EXTENSIONS = {
    ".hdr", ".raw", ".bin",
    ".b2nb", ".b2nd",
    ".bip", ".bil", ".bsq",
    ".npy", ".npz", ".h5",
}

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATASETS_DIR           = PROJECT_ROOT / "datasets"
OUTPUT_DIR             = PROJECT_ROOT / "outputs"
TRAINED_MODELS_DIR     = OUTPUT_DIR  / "trained_models"
EVAL_METRICS_DIR       = OUTPUT_DIR  / "evaluation_metrics"
PREDICTION_RESULTS_DIR = OUTPUT_DIR  / "prediction_results"
HEATMAPS_DIR           = OUTPUT_DIR  / "heatmaps"
LOGS_DIR               = PROJECT_ROOT / "logs"

# Auto-create all output directories on first import so training/inference
# never fail with FileNotFoundError on the very first run.
for _d in (TRAINED_MODELS_DIR, EVAL_METRICS_DIR,
           PREDICTION_RESULTS_DIR, HEATMAPS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Dataset roots (override via environment variables)
# Default paths match the actual dataset layout on the development machine.
# To override, set env vars before running:
#   set HSI_TUMOR_DATASET=E:\umasagar\datasets\tumor_dataset\HistologyHSI-GB
#   set HSI_PERFUSION_DATASET=E:\umasagar\datasets\perfusion_dataset\SPECTRALPACA
import os

# Detect if running on Windows with E: drive datasets
_WIN_TUMOR      = Path("E:/umasagar/datasets/tumor_dataset/HistologyHSI-GB")
_WIN_PERFUSION  = Path("E:/umasagar/datasets/perfusion_dataset/SPECTRALPACA")

TUMOR_DATASET_ROOT = Path(os.getenv(
    "HSI_TUMOR_DATASET",
    str(_WIN_TUMOR) if _WIN_TUMOR.exists() else str(DATASETS_DIR / "tumor_dataset" / "HistologyHSI-GB")
))
PERFUSION_DATASET_ROOT = Path(os.getenv(
    "HSI_PERFUSION_DATASET",
    str(_WIN_PERFUSION) if _WIN_PERFUSION.exists() else str(DATASETS_DIR / "perfusion_dataset" / "SPECTRALPACA")
))

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_FILE   = LOGS_DIR / "hsi_framework.log"

# ─── Model versioning ──────────────────────────────────────────────────────────
# When True, saved model files include a timestamp suffix:
#   e.g. tumor_rf_20250306_143022.pkl
# The canonical names (tumor_rf.pkl) are always updated via symlink/copy.
ENABLE_MODEL_VERSIONING = True

# ─── Legacy / convenience aliases ─────────────────────────────────────────────
MODEL_OUTPUT  = TRAINED_MODELS_DIR   # alias: models are saved here
DATASET_PATH  = DATASETS_DIR         # alias: root dataset directory

# ─── Convenience bundle ────────────────────────────────────────────────────────
CFG = {
    "patch_size":              PATCH_SIZE,
    "stride":                  STRIDE,
    "stride_perf":             STRIDE_PERF,
    "max_patches":             MAX_PATCHES,
    "n_informative_bands":     N_INFORMATIVE_BANDS,
    "rf_trees":                RF_TREES,
    "rf_max_depth":            RF_MAX_DEPTH,
    "svm_kernel":              SVM_KERNEL,
    "svm_c_tumor":             SVM_C_TUMOR,
    "svm_c_perf":              SVM_C_PERF,
    "n_pca_tumor":             N_PCA_TUMOR,
    "n_pca_perf":              N_PCA_PERF,
    "use_majority_vote":       USE_MAJORITY_VOTE,
    "large_cube_elements":     LARGE_CUBE_ELEMENTS,
    "min_bands":               MIN_BANDS,
    "patch_batch_size":        PATCH_BATCH_SIZE,
    "allowed_upload_exts":     ALLOWED_UPLOAD_EXTENSIONS,
}