"""
ENVI Hyperspectral Image Loader — Unified HSI Medical Framework
===============================================================

Safety validations enforced (per PDF spec §6):
  1. HDR file must end with .hdr
  2. RAW file must NOT be passed as the HDR argument
  3. Filenames must match (same stem)
  4. NaN / Inf check after loading
  5. Minimum spectral bands check (cube.shape[2] >= MIN_BANDS)
  6. Large cube warning when cube.size > LARGE_CUBE_ELEMENTS
  7. envi.open(hdr_path, raw_path) is used — no alternative loader

Normalisation note:
  This loader returns raw DN values as float32. Radiometric normalisation
  (global min-max or dark/white reference correction) is applied downstream
  in ml_pipeline/preprocessing/radiometric.py — NOT here.

Returns cube with shape (H, W, B) float32.
"""

import io
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Import threshold from config (fallback if config not yet set up)
try:
    from config.config import LARGE_CUBE_ELEMENTS
except ImportError:
    LARGE_CUBE_ELEMENTS = int(2e9)


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_hdr_path(hdr_path: Path) -> None:
    """Raise ValueError if hdr_path does not look like a proper .hdr file."""
    if hdr_path.suffix.lower() != ".hdr":
        raise ValueError(
            f"HDR path must end with '.hdr', got: '{hdr_path}'. "
            "Ensure you are passing the header file, not the raw binary."
        )
    # Guard: user must not accidentally pass a raw binary as the HDR
    for raw_ext in (".raw", ".bin", ".bip", ".bil", ".bsq"):
        if hdr_path.suffix.lower() == raw_ext:
            raise ValueError(
                f"Raw binary file ('{hdr_path.name}') was passed as the HDR argument. "
                "Swap the arguments: load_hyperspectral_image(hdr_path, raw_path)."
            )


def _validate_filename_match(hdr_path: Path, raw_path: Path) -> None:
    """Warn if hdr and raw stems differ (common user mistake)."""
    if hdr_path.stem.lower() != raw_path.stem.lower():
        warnings.warn(
            f"HDR stem '{hdr_path.stem}' ≠ RAW stem '{raw_path.stem}'. "
            "Files may not belong to the same dataset.",
            UserWarning,
            stacklevel=3,
        )
        logger.warning(
            "Filename mismatch: hdr='%s', raw='%s'", hdr_path.name, raw_path.name
        )


def _warn_large_cube(cube: np.ndarray) -> None:
    if cube.size > LARGE_CUBE_ELEMENTS:
        logger.warning(
            "Large cube detected: %d elements (shape %s). "
            "Processing may be slow and memory-intensive.",
            cube.size, cube.shape,
        )


def _validate_cube_content(cube: np.ndarray, source: str = "") -> None:
    """Raise RuntimeError for NaN / Inf values or insufficient spectral bands."""
    if not np.isfinite(cube).all():
        n_nan = int(np.isnan(cube).sum())
        n_inf = int(np.isinf(cube).sum())
        raise RuntimeError(
            f"Hyperspectral cube from '{source}' contains invalid values: "
            f"{n_nan} NaN, {n_inf} Inf. File may be corrupt or truncated."
        )
    # Minimum bands check (spec Final Changes §8)
    try:
        from config.config import MIN_BANDS
    except ImportError:
        MIN_BANDS = 10
    if cube.ndim == 3 and cube.shape[2] < MIN_BANDS:
        raise ValueError(
            f"Hyperspectral cube from '{source}' has only {cube.shape[2]} spectral "
            f"bands (minimum required: {MIN_BANDS}). "
            "The file may be a greyscale or RGB image, not a hyperspectral cube."
        )


def _normalise_cube_shape(cube: np.ndarray) -> np.ndarray:
    """Ensure output is always (H, W, B)."""
    if cube.ndim == 2:
        cube = cube[:, :, np.newaxis]
    elif cube.ndim == 3:
        # spectral library sometimes returns (B, H, W) when first dim is bands
        if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
            cube = np.transpose(cube, (1, 2, 0))
    else:
        raise ValueError(
            f"Unexpected cube ndim={cube.ndim} (shape={cube.shape}); "
            "expected 2-D or 3-D array."
        )
    H, W, B = cube.shape
    if H <= 0 or W <= 0 or B <= 0:
        raise ValueError(f"Non-positive cube dimension: {cube.shape}")
    return cube


# ──────────────────────────────────────────────────────────────────────────────
# Primary loader: ENVI .hdr + raw
# ──────────────────────────────────────────────────────────────────────────────

def load_hyperspectral_image(
    hdr_path: Union[str, Path],
    raw_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Load an ENVI hyperspectral file pair (.hdr + raw binary).

    Parameters
    ----------
    hdr_path : path to the .hdr header file
    raw_path : path to the raw binary data file (optional — inferred if omitted)

    Returns
    -------
    (H, W, B) float32 NumPy array

    Safety checks
    -------------
    • hdr_path must end with .hdr
    • raw_path must NOT be a .hdr file passed by mistake
    • Filename stems are compared; mismatch triggers a warning
    • NaN/Inf values raise RuntimeError
    • Cubes > LARGE_CUBE_ELEMENTS trigger a logged warning
    """
    hdr_path = Path(hdr_path)

    # ── Validation ────────────────────────────────────────────────────────────
    _validate_hdr_path(hdr_path)

    if not hdr_path.exists():
        raise FileNotFoundError(f"HDR file not found: {hdr_path}")

    # Resolve raw_path
    if raw_path is not None:
        raw_path = Path(raw_path)
        # Guard: user must not swap hdr/raw arguments
        if raw_path.suffix.lower() == ".hdr":
            raise ValueError(
                f"raw_path '{raw_path}' appears to be a .hdr file. "
                "Arguments are load_hyperspectral_image(hdr_path, raw_path)."
            )
    else:
        raw_path = hdr_path.with_suffix("")          # ENVI convention
        if not raw_path.exists():
            raw_path = hdr_path.with_suffix(".raw")  # fallback

    if not raw_path.exists():
        raise FileNotFoundError(
            f"RAW binary file not found for '{hdr_path.name}'. "
            f"Expected: '{raw_path}'. "
            "Ensure the raw data file is in the same directory as the .hdr."
        )

    _validate_filename_match(hdr_path, raw_path)

    # ── Load via spectral.envi ─────────────────────────────────────────────────
    try:
        from spectral import envi
    except ImportError:
        raise ImportError(
            "spectral library is required. Install with: pip install spectral"
        )

    logger.info("Loading ENVI cube  HDR='%s'  RAW='%s'", hdr_path, raw_path)

    try:
        img  = envi.open(str(hdr_path), str(raw_path))   # mandatory call per spec §6
        cube = np.array(img.load(), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"spectral.envi.open failed for '{hdr_path}': {exc}"
        ) from exc

    # ── Post-load normalisation & validation ──────────────────────────────────
    cube = _normalise_cube_shape(cube)
    _warn_large_cube(cube)
    _validate_cube_content(cube, source=str(hdr_path))

    logger.info(
        "Cube loaded successfully — shape (H,W,B)=%s  dtype=%s  "
        "min=%.4f  max=%.4f",
        cube.shape, cube.dtype, float(cube.min()), float(cube.max())
    )
    return cube


# ──────────────────────────────────────────────────────────────────────────────
# Byte-stream loader (used by FastAPI route for in-memory uploads)
# ──────────────────────────────────────────────────────────────────────────────

def load_hyperspectral_from_bytes(
    data:       bytes,
    ext:        str,
    filename:   str,
    hdr_bytes:  Optional[bytes] = None,
) -> np.ndarray:
    """
    Load a hyperspectral cube from raw bytes.

    Supports: .npy  .npz  .h5  .hdr+raw (via temp directory)

    Parameters
    ----------
    data      : raw file bytes
    ext       : file extension (e.g. '.npy')
    filename  : original filename (for stem matching)
    hdr_bytes : if ext='.hdr', the companion raw bytes go here

    Returns
    -------
    (H, W, B) float32 array
    """
    ext = ext.lower()

    if ext == ".npy":
        cube = np.load(io.BytesIO(data)).astype(np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        _warn_large_cube(cube)
        _validate_cube_content(cube, source=filename)
        return cube

    if ext == ".npz":
        npz  = np.load(io.BytesIO(data))
        key  = list(npz.files)[0]
        cube = npz[key].astype(np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        _warn_large_cube(cube)
        _validate_cube_content(cube, source=filename)
        return cube

    if ext == ".h5":
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")
        with h5py.File(io.BytesIO(data), "r") as f:
            key  = list(f.keys())[0]
            cube = np.asarray(f[key], dtype=np.float32)
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        _warn_large_cube(cube)
        _validate_cube_content(cube, source=filename)
        return cube

    if ext in (".hdr", ".raw", ".bin", ".bip", ".bil", ".bsq", ""):
        if hdr_bytes is None:
            raise ValueError(
                "ENVI format requires both the .hdr header bytes and the "
                "raw binary bytes. Pass hdr_bytes= for the header."
            )
        with tempfile.TemporaryDirectory(prefix="hsi_") as tmp:
            tmp     = Path(tmp)
            stem    = Path(filename).stem
            hdr_p   = tmp / f"{stem}.hdr"
            raw_p   = tmp / stem
            hdr_p.write_bytes(hdr_bytes)
            raw_p.write_bytes(data)
            return load_hyperspectral_image(hdr_p, raw_p)

    raise ValueError(
        f"Unsupported file extension: '{ext}'. "
        "Use .npy / .npz / .h5 / .hdr / .b2nd"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def get_cube_shape(cube: np.ndarray) -> Tuple[int, int, int]:
    if cube.ndim == 2:
        return cube.shape[0], cube.shape[1], 1
    return cube.shape[0], cube.shape[1], cube.shape[2]
