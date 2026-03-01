"""
Radiometric Preprocessing for Hyperspectral Medical Data.

Applies dark reference subtraction and white reference normalization
to convert raw DN values to reflectance [0,1].
"""
import numpy as np
from pathlib import Path


def load_reference(path: str) -> np.ndarray:
    """Load a reference image (dark or white) from .npy file."""
    return np.load(path).astype(np.float32)


def apply_radiometric_correction(
    raw_cube: np.ndarray,
    dark_ref: np.ndarray = None,
    white_ref: np.ndarray = None
) -> np.ndarray:
    """
    Convert raw HSI cube to reflectance using:
        Reflectance = (raw - dark) / (white - dark + eps)

    Args:
        raw_cube:  (H, W, B) float32 array of raw DN values
        dark_ref:  (B,) or (H, W, B) dark reference frame
        white_ref: (B,) or (H, W, B) white reference frame

    Returns:
        reflectance: (H, W, B) array clipped to [0, 1]
    """
    eps = 1e-8
    cube = raw_cube.astype(np.float64)

    if dark_ref is None:
        dark = 0.0
    else:
        dark = dark_ref.astype(np.float64)
        if dark.ndim == 1:
            dark = dark[np.newaxis, np.newaxis, :]  # broadcast over H, W

    if white_ref is None:
        white = cube.max()
    else:
        white = white_ref.astype(np.float64)
        if white.ndim == 1:
            white = white[np.newaxis, np.newaxis, :]

    # Corrected = (Raw - Dark) / (White - Dark) with division-by-zero protection
    reflectance = (cube - dark) / (white - dark + eps)
    return np.clip(reflectance, 0.0, 1.0).astype(np.float32)


def load_envi_cube(hdr_path: str) -> np.ndarray:
    """
    Load ENVI hyperspectral file (.hdr + raw binary).
    Parses header metadata and reads the binary data file.
    """
    hdr_path = Path(hdr_path)
    raw_path = hdr_path.with_suffix("")

    params = _parse_envi_header(hdr_path)
    lines = int(params.get("lines", 32))
    samples = int(params.get("samples", 32))
    bands = int(params.get("bands", 16))
    dtype_code = int(params.get("data type", 4))
    interleave = params.get("interleave", "bip").lower()

    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16}
    dtype = dtype_map.get(dtype_code, np.float32)

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
    """Parse ENVI .hdr file into a key-value dictionary."""
    params = {}
    with open(hdr_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, _, val = line.partition("=")
                params[key.strip().lower()] = val.strip().strip("{}")
    return params


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """Min-max normalize a cube to [0, 1] independently per band."""
    H, W, B = cube.shape
    out = np.zeros_like(cube, dtype=np.float32)
    for b in range(B):
        band = cube[:, :, b]
        mn, mx = band.min(), band.max()
        if mx - mn > 1e-8:
            out[:, :, b] = (band - mn) / (mx - mn)
    return out


def extract_patches(cube: np.ndarray, patch_size: int = 32, stride: int = 16):
    """
    Extract overlapping patches from a hyperspectral cube.

    Args:
        cube:       (H, W, B) float32
        patch_size: spatial size of each patch (default 32)
        stride:     stride between patches (default 16)

    Yields:
        (row_start, col_start, patch) tuples
    """
    H, W, B = cube.shape
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = cube[r:r + patch_size, c:c + patch_size, :]
            yield (r, c, patch)


def remove_bad_bands(cube: np.ndarray, noise_threshold: float = 0.01) -> np.ndarray:
    """
    Remove spectral bands with near-zero variance (noisy or dead bands).

    Args:
        cube:             (H, W, B)
        noise_threshold:  bands with variance below this are removed

    Returns:
        Filtered cube with low-variance bands removed
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    band_var = flat.var(axis=0)
    good_bands = band_var > noise_threshold
    if good_bands.sum() == 0:
        return cube  # keep all if everything is noisy
    return cube[:, :, good_bands]
