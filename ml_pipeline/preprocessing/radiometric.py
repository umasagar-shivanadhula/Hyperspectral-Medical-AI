"""
Radiometric Preprocessing — Unified HSI Medical Framework
==========================================================

Safety rules (per PDF spec §6):
  • apply_radiometric_correction applies dark/white correction when BOTH
    references are provided; otherwise falls back to global min-max
    normalisation:  out = (cube - min) / (max - min + eps)
  • All outputs clipped to [0, 1] float32.

Public API
----------
apply_radiometric_correction(cube, dark_ref, white_ref)
normalize_global(cube)
normalize_cube(cube)
load_reference(path)
extract_patches(cube, patch_size, stride)   ← thin delegate to patch_extraction
load_envi_cube(hdr_path)                    ← legacy fallback loader
remove_bad_bands(cube, noise_threshold)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reference loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_reference(path: str) -> np.ndarray:
    """Load a dark or white reference cube from a .npy file."""
    return np.load(path).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Radiometric correction
# ──────────────────────────────────────────────────────────────────────────────

def apply_radiometric_correction(
    raw_cube:  np.ndarray,
    dark_ref:  Optional[np.ndarray] = None,
    white_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert raw DN values to reflectance [0, 1].

    Formula:  reflectance = (raw - dark) / (white - dark + eps)

    Safety rule (§6): correction is applied ONLY when BOTH dark and white
    references are provided. If either is missing, a per-band max normalisation
    is used instead — this matches the training pipeline behaviour when no
    calibration data is available.

    Parameters
    ----------
    raw_cube  : (H, W, B) float array
    dark_ref  : dark reference  — (B,) or (H, W, B) or None
    white_ref : white reference — (B,) or (H, W, B) or None

    Returns
    -------
    (H, W, B) float32, clipped to [0, 1]
    """
    eps  = 1e-8
    cube = np.asarray(raw_cube, dtype=np.float64)

    # ── Full correction (both references available) ────────────────────────
    if dark_ref is not None and white_ref is not None:
        dark  = np.asarray(dark_ref,  dtype=np.float64)
        white = np.asarray(white_ref, dtype=np.float64)

        # Broadcast per-band vectors
        if dark.ndim  == 1: dark  = dark [np.newaxis, np.newaxis, :]
        if white.ndim == 1: white = white[np.newaxis, np.newaxis, :]

        reflectance = (cube - dark) / (white - dark + eps)
        logger.debug("Radiometric correction applied (dark + white refs).")

    # ── Partial / no references → global min-max normalisation ───────────
    else:
        if dark_ref is not None or white_ref is not None:
            logger.warning(
                "Only one reference (dark=%s, white=%s) was provided. "
                "Full correction requires both. Falling back to global "
                "min-max normalisation.",
                dark_ref is not None, white_ref is not None,
            )
        else:
            logger.debug("No references provided — using global min-max normalisation.")

        # Global min-max (spec Final Changes §2):
        #   out = (cube - cube.min()) / (cube.max() - cube.min() + eps)
        mn         = cube.min()
        mx         = cube.max()
        reflectance = (cube - mn) / (mx - mn + eps)

    result = np.clip(reflectance, 0.0, 1.0).astype(np.float32)
    logger.debug(
        "Radiometric correction done — shape=%s  min=%.4f  max=%.4f",
        result.shape, float(result.min()), float(result.max())
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Global min-max normalisation (spec Final Changes §2)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_global(cube: np.ndarray) -> np.ndarray:
    """
    Global (cube-wide) min–max normalisation to [0, 1].

    Formula:  out = (cube - cube.min()) / (cube.max() - cube.min() + eps)

    This ensures spectral consistency across all bands simultaneously,
    as required by the Final Changes specification §2.
    Use this when no dark/white calibration references are available.
    """
    mn  = float(cube.min())
    mx  = float(cube.max())
    eps = 1e-8
    out = (cube.astype(np.float64) - mn) / (mx - mn + eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """Per-band min–max normalisation to [0, 1]."""
    H, W, B = cube.shape
    out = np.zeros_like(cube, dtype=np.float32)
    for b in range(B):
        band  = cube[:, :, b]
        mn, mx = float(band.min()), float(band.max())
        if mx - mn > 1e-8:
            out[:, :, b] = (band - mn) / (mx - mn)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Patch extraction delegate (kept here for backward-compat imports)
# ──────────────────────────────────────────────────────────────────────────────

def extract_patches(cube: np.ndarray, patch_size: int = 32, stride: int = 16):
    """
    Delegate to ml_pipeline.preprocessing.patch_extraction.extract_patches.

    Yields (row_start, col_start, patch) triples.
    Patches are 3-D (patch_size, patch_size, B) — never flattened.
    """
    from ml_pipeline.preprocessing.patch_extraction import extract_patches as _ep
    yield from _ep(cube, patch_size=patch_size, stride=stride)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy ENVI loader (used by training scripts as fallback)
# ──────────────────────────────────────────────────────────────────────────────

def load_envi_cube(hdr_path: str) -> np.ndarray:
    """
    Parse ENVI .hdr header manually and load binary data.
    Used as fallback when spectral library is unavailable.
    """
    hdr_path = Path(hdr_path)
    raw_path = hdr_path.with_suffix("")

    params   = _parse_envi_header(hdr_path)
    lines    = int(params.get("lines",     32))
    samples  = int(params.get("samples",   32))
    bands    = int(params.get("bands",     16))
    dtype_c  = int(params.get("data type",  4))
    interleave = params.get("interleave", "bip").lower()

    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32,
                 4: np.float32, 5: np.float64, 12: np.uint16}
    dtype    = dtype_map.get(dtype_c, np.float32)

    if not raw_path.exists():
        raise FileNotFoundError(f"Binary data file not found: {raw_path}")

    data = np.fromfile(str(raw_path), dtype=dtype)

    if interleave == "bip":
        cube = data.reshape(lines, samples, bands)
    elif interleave == "bsq":
        cube = data.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = data.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:
        cube = data.reshape(lines, samples, bands)

    return cube.astype(np.float32)


def _parse_envi_header(hdr_path: Path) -> dict:
    params = {}
    with open(hdr_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, _, val = line.partition("=")
                params[key.strip().lower()] = val.strip().strip("{}")
    return params


def remove_bad_bands(cube: np.ndarray, noise_threshold: float = 0.01) -> np.ndarray:
    """
    Remove spectral bands with near-zero variance (noisy / dead bands).

    Parameters
    ----------
    cube            : (H, W, B) float32 array
    noise_threshold : bands with variance below this are considered dead

    Returns
    -------
    (H, W, B') float32 array with bad bands removed.
    Returns the original cube unchanged if it is None or all bands are bad.
    """
    if cube is None:
        return cube

    H, W, B = cube.shape
    flat     = cube.reshape(-1, B)
    band_var = flat.var(axis=0)
    good     = band_var > noise_threshold

    if good.sum() == 0:
        # All bands would be removed — keep original to avoid empty cube
        logger.warning(
            "remove_bad_bands: all %d bands fall below threshold=%.4f; "
            "returning original cube unchanged.",
            B, noise_threshold,
        )
        return cube

    selected = cube[:, :, good]
    logger.debug(
        "remove_bad_bands: kept %d / %d bands (threshold=%.4f)",
        int(good.sum()), B, noise_threshold,
    )
    return selected
