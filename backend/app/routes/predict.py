"""
FastAPI Prediction Routes — Unified HSI Medical Framework
==========================================================

Endpoints:
  POST /predict/tumor                — single-file upload → full tumor prediction
  POST /predict/perfusion            — single-file upload → full perfusion prediction
  GET  /predict/evaluation/tumor     — last training evaluation metrics JSON
  GET  /predict/evaluation/perfusion — last training evaluation metrics JSON

Upload validation (spec §6):
  • Only allowed extensions accepted:
      .hdr .raw .bin .b2nb .b2nd .bip .bil .bsq .npy .npz .h5
  • Unknown extensions → HTTP 400

Cube safety (applied inside pipeline):
  • NaN/Inf         → HTTP 422
  • Min bands < 10  → HTTP 422
  • Large cube warning logged automatically

Model safety (applied inside pipeline):
  • Missing model files → HTTP 503 with clear message

Pipeline delegation:
  • All ML logic lives exclusively in ml_pipeline/prediction/predict.py.
  • Routes call run_pipeline_on_cube(cube, task) — no duplication.
"""

import io
import json
import logging
import uuid
from pathlib import Path

import shutil
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config.config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    EVAL_METRICS_DIR,
    TRAINED_MODELS_DIR,
    PREDICTION_RESULTS_DIR,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
UPLOAD_DIR    = _PROJECT_ROOT / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _load_cube_from_hdr_pair(hdr_file: UploadFile, raw_file: UploadFile) -> "np.ndarray":
    """
    Load an ENVI hyperspectral cube from an .hdr + .raw file pair.

    ENVI format stores metadata in a plain-text .hdr header and pixel
    data in a companion binary file (commonly named <stem>.raw or just
    <stem> with no extension).  Both must be present on disk for
    spectral.envi.open() to succeed.
    """
    hdr_data = hdr_file.file.read()
    raw_data = raw_file.file.read()

    uid      = str(uuid.uuid4())
    stem     = Path(hdr_file.filename or "cube").stem
    tmp_dir  = UPLOAD_DIR / uid
    tmp_dir.mkdir(parents=True, exist_ok=True)

    hdr_path = tmp_dir / f"{stem}.hdr"
    raw_path = tmp_dir / stem          # no extension — matches ENVI convention

    hdr_path.write_bytes(hdr_data)
    raw_path.write_bytes(raw_data)

    logger.info(
        "[HDR-PAIR] hdr=%s (%d B)  raw=%s (%d B)",
        hdr_path.name, len(hdr_data), raw_path.name, len(raw_data),
    )

    try:
        from ml_pipeline.data_loader.hsi_loader import load_hyperspectral_image
        cube = load_hyperspectral_image(str(hdr_path), str(raw_path))
        return cube
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Upload extension validation (spec §6)
# ──────────────────────────────────────────────────────────────────────────────

def _validate_extension(filename: str) -> str:
    """Return the lowercase extension or raise HTTP 400 for disallowed types."""
    ext = Path(filename).suffix.lower() if filename else ""
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File extension '{ext}' is not allowed. "
                f"Accepted formats: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
            ),
        )
    return ext


# ──────────────────────────────────────────────────────────────────────────────
# Cube loader from UploadFile
# ──────────────────────────────────────────────────────────────────────────────

def _load_cube_from_upload(file: UploadFile) -> np.ndarray:
    """
    Load a (H,W,B) float32 cube from a single UploadFile.
    Validates extension before loading.
    """
    filename = file.filename or "upload.bin"
    ext      = _validate_extension(filename)
    data     = file.file.read()

    logger.info("[UPLOAD] file='%s'  ext='%s'  size=%d bytes", filename, ext, len(data))

    # ── NumPy ─────────────────────────────────────────────────────────────
    if ext == ".npy":
        cube = np.load(io.BytesIO(data)).astype(np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        return cube

    if ext == ".npz":
        npz  = np.load(io.BytesIO(data))
        key  = list(npz.files)[0]
        cube = npz[key].astype(np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        return cube

    # ── HDF5 ──────────────────────────────────────────────────────────────
    if ext == ".h5":
        try:
            import h5py
        except ImportError:
            raise HTTPException(status_code=500, detail="h5py not installed: pip install h5py")
        with h5py.File(io.BytesIO(data), "r") as f:
            key  = list(f.keys())[0]
            cube = np.asarray(f[key], dtype=np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        return cube

    # ── Blosc2 B2ND ───────────────────────────────────────────────────────
    if ext in (".b2nd", ".b2nb"):
        uid      = str(uuid.uuid4())
        tmp_path = UPLOAD_DIR / f"{uid}.b2nd"
        tmp_path.write_bytes(data)
        try:
            from ml_pipeline.data_loader.b2nd_loader import load_b2nd_cube
            nda  = load_b2nd_cube(str(tmp_path))

            # ── Chunked sampling — never load full array into RAM ──────────
            # Shape possibilities:
            #   (H, W, B)         → 3-D spatial cube, use directly
            #   (T, H, W, B)      → 4-D time series (SPECTRALPACA), sample T
            #   (T, H, W)         → 3-D time+spatial, sample T
            shape = nda.shape
            ndim  = len(shape)

            MAX_FRAMES   = 64     # max time frames to read
            MAX_SPATIAL  = 256    # max H and W (downsampled)
            MAX_BANDS    = 16     # keep up to 16 spectral bands

            logger.info("[B2ND] Raw shape %s  dtype %s", shape, nda.dtype)

            if ndim == 4:
                T, H, W, B = shape
                # Sample frames uniformly
                frame_idx = np.linspace(0, T - 1, min(MAX_FRAMES, T), dtype=int)
                # Spatial stride to cap H/W
                h_stride  = max(1, H // MAX_SPATIAL)
                w_stride  = max(1, W // MAX_SPATIAL)
                b_end     = min(B, MAX_BANDS)

                slices = []
                for fi in frame_idx:
                    # Read one slice at a time — blosc2 decompresses chunk-by-chunk
                    sl = np.array(nda[fi, ::h_stride, ::w_stride, :b_end], dtype=np.float32)
                    slices.append(sl)
                cube = np.stack(slices, axis=0).mean(axis=0)   # → (H', W', B')

            elif ndim == 3 and shape[0] > 512:
                # Likely (T, H, W) — large first dim = time
                T, H, W = shape
                frame_idx = np.linspace(0, T - 1, min(MAX_FRAMES, T), dtype=int)
                slices = [np.array(nda[fi, :, :], dtype=np.float32) for fi in frame_idx]
                arr    = np.stack(slices, axis=0).mean(axis=0)   # → (H, W)
                cube   = arr[:, :, np.newaxis]                   # → (H, W, 1)

            else:
                # Normal (H, W, B) — safe to load directly if not huge
                elem = np.prod(shape)
                if elem > 256 * 256 * 32:
                    h_stride = max(1, shape[0] // MAX_SPATIAL)
                    w_stride = max(1, shape[1] // MAX_SPATIAL) if ndim >= 2 else 1
                    cube = np.array(nda[::h_stride, ::w_stride], dtype=np.float32)
                else:
                    cube = np.array(nda, dtype=np.float32)
                if cube.ndim == 2:
                    cube = cube[:, :, np.newaxis]

            logger.info("[B2ND] Sampled cube shape %s", cube.shape)
            return cube
        finally:
            try: tmp_path.unlink()
            except: pass

    # ── ENVI .hdr — needs companion .raw binary ──────────────────────────
    # .hdr is only the header; pixel data lives in the companion .raw file.
    # The endpoint (_load_cube_from_hdr_pair) handles the pair upload.
    # If someone hits this path they sent .hdr without .raw — tell them.
    if ext == ".hdr":
        raise HTTPException(
            status_code=400,
            detail=(
                "ENVI .hdr files require the companion .raw data file. "
                "Please upload BOTH files together: select your .hdr file and "
                "then also select the matching .raw binary file."
            ),
        )

    # ── Raw binary fallback ───────────────────────────────────────────────
    raw   = np.frombuffer(data, dtype=np.float32)
    total = raw.size
    target = 32 * 32 * 16
    if total >= target:
        cube = raw[:target].reshape(32, 32, 16)
    else:
        padded = np.zeros(target, dtype=np.float32)
        padded[:total] = raw
        cube = padded.reshape(32, 32, 16)
    logger.warning("[UPLOAD] Loaded as raw float32 fallback — shape %s", cube.shape)
    return cube


# ──────────────────────────────────────────────────────────────────────────────
# ML pipeline delegation
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(cube: np.ndarray, task: str) -> dict:
    """
    Delegate inference to ml_pipeline/prediction/predict.py.

    All ML logic lives there — routes contain zero pipeline code.
    """
    from ml_pipeline.prediction.predict import run_pipeline_on_cube
    return run_pipeline_on_cube(cube, task)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/tumor")
async def tumor_predict(
    file:     UploadFile          = File(...),
    raw_file: Optional[UploadFile] = File(None),
):
    """
    Upload one hyperspectral file → full tumor prediction JSON.

    For ENVI files upload BOTH:
      FormData { file: <name.hdr>, raw_file: <name.raw> }
    For all other formats:
      FormData { file: <name.npy/.npz/.h5/...> }
    """
    try:
        ext = Path(file.filename or "").suffix.lower()
        if ext == ".hdr":
            if raw_file is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "ENVI .hdr files require the companion .raw data file. "
                        "Please upload BOTH files: the .hdr header AND the .raw binary."
                    ),
                )
            cube = _load_cube_from_hdr_pair(file, raw_file)
        else:
            cube = _load_cube_from_upload(file)
        result = _run_pipeline(cube, "tumor")
        return result
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("[TUMOR] Missing models: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.error("[TUMOR] Validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[TUMOR] Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/perfusion")
async def perfusion_predict(
    file:     UploadFile          = File(...),
    raw_file: Optional[UploadFile] = File(None),
):
    """
    Upload one hyperspectral file → full perfusion prediction JSON.

    For ENVI files upload BOTH:
      FormData { file: <name.hdr>, raw_file: <name.raw> }
    For all other formats:
      FormData { file: <name.npy/.npz/.h5/...> }
    """
    try:
        ext = Path(file.filename or "").suffix.lower()
        if ext == ".hdr":
            if raw_file is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "ENVI .hdr files require the companion .raw data file. "
                        "Please upload BOTH files: the .hdr header AND the .raw binary."
                    ),
                )
            cube = _load_cube_from_hdr_pair(file, raw_file)
        else:
            cube = _load_cube_from_upload(file)
        result = _run_pipeline(cube, "perfusion")
        return result
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("[PERFUSION] Missing models: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.error("[PERFUSION] Validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[PERFUSION] Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/evaluation/tumor")
async def evaluation_tumor():
    """
    Return evaluation metrics saved by train_tumor.py.
    Frontend expects: { fusion: { accuracy, f1_score, confusion_matrix,
                                   confusion_matrix_labels }, classes }
    """
    path = EVAL_METRICS_DIR / "tumor_metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation data found. "
                "Run: python ml_pipeline/training/train_tumor.py"
            ),
        )
    with open(path, "r") as f:
        return JSONResponse(content=json.load(f))


@router.get("/evaluation/perfusion")
async def evaluation_perfusion():
    """
    Return evaluation metrics saved by train_perfusion.py.
    Frontend expects: { fusion: { accuracy, f1_score, confusion_matrix,
                                   confusion_matrix_labels }, classes }
    """
    path = EVAL_METRICS_DIR / "perfusion_metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation data found. "
                "Run: python ml_pipeline/training/train_perfusion.py"
            ),
        )
    with open(path, "r") as f:
        return JSONResponse(content=json.load(f))