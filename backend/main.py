"""
Unified HSI Medical Framework — FastAPI Backend
================================================
Run from project root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import logging.handlers
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Structured logging setup ──────────────────────────────────────────────────
from config.config import LOG_FORMAT, LOG_LEVEL, LOG_FILE, LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Console handler
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter(LOG_FORMAT))

# Rotating file handler (10 MB × 5 backups)
_file = logging.handlers.RotatingFileHandler(
    str(LOG_FILE), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    handlers=[_console, _file])

logger = logging.getLogger(__name__)
logger.info("HSI Medical Framework starting — project root: %s", PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.app.routes.predict import router as predict_router

app = FastAPI(
    title       = "Unified HSI Medical Framework API",
    description = "Spectral-Spatial Multi-Classifier Hyperspectral Medical Image Analysis",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(predict_router, prefix="/predict", tags=["Prediction"])


@app.get("/")
def root():
    return {
        "project":     "Unified HSI Medical Framework",
        "version":     "2.0.0",
        "description": "Spectral-Spatial RF+SVM Softmax Fusion for HSI Medical Image Analysis",
        "endpoints": {
            "tumor_prediction":      "/predict/tumor",
            "perfusion_prediction":  "/predict/perfusion",
            "tumor_evaluation":      "/predict/evaluation/tumor",
            "perfusion_evaluation":  "/predict/evaluation/perfusion",
            "health":                "/health",
            "docs":                  "/docs",
        },
        "status": "running",
    }


@app.get("/health")
def health():
    """Health check endpoint (spec §9.6)."""
    from config.config import TRAINED_MODELS_DIR
    models = {
        "tumor_rf":         (TRAINED_MODELS_DIR / "tumor_rf.pkl").exists(),
        "tumor_svm":        (TRAINED_MODELS_DIR / "tumor_svm.pkl").exists(),
        "perfusion_rf":     (TRAINED_MODELS_DIR / "perfusion_rf.pkl").exists(),
        "perfusion_svm":    (TRAINED_MODELS_DIR / "perfusion_svm.pkl").exists(),
    }
    return {
        "status":        "ok",
        "service":       "HSI Medical Backend",
        "models_loaded": models,
        "all_models_ready": all(models.values()),
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
