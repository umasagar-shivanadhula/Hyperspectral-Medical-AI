"""
Prediction API routes for Perfusion and Tumor detection.
"""
import json
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()
_service = None


def get_service():
    """Lazy-init service (imports ml_pipeline on first predict)."""
    global _service
    if _service is None:
        from app.services.ml_service import HSIPipelineService
        _service = HSIPipelineService()
    return _service

EVAL_DIR = Path(__file__).resolve().parents[3] / "outputs" / "evaluation_metrics"


@router.post("/perfusion")
async def predict_perfusion(
    file: UploadFile = File(...),
    file_hdr: UploadFile = File(None),
):
    """
    Predict tissue perfusion state from hyperspectral data.
    Accepts .npy, .npz, .h5, or ENVI .hdr + .raw (upload .raw as file and .hdr as file_hdr).
    Returns prediction, confidence, rf_prediction, svm_prediction, classifier_probabilities.
    """
    allowed = {".npy", ".npz", ".hdr", ".h5", ".bin", ".raw"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use .npy, .npz, .h5, or .hdr+.raw",
        )
    contents = await file.read()
    hdr_bytes = None
    if file_hdr and file_hdr.filename:
        hdr_bytes = await file_hdr.read()
    try:
        service = get_service()
        hsi_cube = service.load_hsi_data(contents, ext, file.filename, hdr_bytes=hdr_bytes)
        result = service.predict_perfusion(hsi_cube)
        return result
    except RuntimeError as e:
        if "not found" in str(e).lower() or "train" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.post("/tumor")
async def predict_tumor(
    file: UploadFile = File(...),
    file_hdr: UploadFile = File(None),
):
    """
    Detect glioblastoma tumor tissue from hyperspectral histology image.
    Accepts .npy, .npz, .h5, or ENVI .hdr + .raw.
    Returns prediction, confidence, rf_prediction, svm_prediction, classifier_probabilities.
    """
    allowed = {".npy", ".npz", ".hdr", ".h5", ".bin", ".raw"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use .npy, .npz, .h5, or .hdr+.raw",
        )
    contents = await file.read()
    hdr_bytes = None
    if file_hdr and file_hdr.filename:
        hdr_bytes = await file_hdr.read()
    try:
        service = get_service()
        hsi_cube = service.load_hsi_data(contents, ext, file.filename, hdr_bytes=hdr_bytes)
        result = service.predict_tumor(hsi_cube)
        return result
    except RuntimeError as e:
        if "not found" in str(e).lower() or "train" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/evaluation/perfusion")
async def get_perfusion_evaluation():
    """Return last saved perfusion evaluation metrics (accuracy, F1, confusion matrix)."""
    path = EVAL_DIR / "perfusion_evaluation.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No perfusion evaluation found. Run train_perfusion.py first.",
        )
    with open(path) as f:
        return json.load(f)


@router.get("/evaluation/tumor")
async def get_tumor_evaluation():
    """Return last saved tumor evaluation metrics (accuracy, F1, confusion matrix)."""
    path = EVAL_DIR / "tumor_evaluation.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No tumor evaluation found. Run train_tumor.py first.",
        )
    with open(path) as f:
        return json.load(f)
