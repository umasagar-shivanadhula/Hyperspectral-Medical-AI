"""
Loader utilities for Blosc2 B2ND (.b2nd) hyperspectral files.

The SPECTRALPACA perfusion dataset is stored as compressed multidimensional
arrays in the B2ND format. This module provides a thin wrapper around
python-blosc2 so that .b2nd cubes can be loaded as NumPy arrays with shape
H × W × B and dtype float32, matching the expectations of the rest of the
pipeline.
"""
from pathlib import Path
from typing import Union

import numpy as np


def _ensure_blosc2():
    try:
        import blosc2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "blosc2 is required to read .b2nd files. "
            "Install it with: pip install blosc2"
        ) from exc
    return blosc2


def load_b2nd_cube(path: Union[str, Path]) -> np.ndarray:
    """
    Load a .b2nd hyperspectral cube as a NumPy array.

    Args:
        path: Path to a .b2nd file on disk.

    Returns:
        cube: Float32 NumPy array with shape (H, W, B).
              If the stored array is 2D, a singleton spectral band
              dimension is added. If the array is 3D but appears to
              be in (B, H, W) order, it is transposed to (H, W, B).
    """
    path = Path(path)
    blosc2 = _ensure_blosc2()

    nda = blosc2.open(str(path), mode="r")
    cube = np.asarray(nda, dtype=np.float32)

    if cube.ndim == 2:
        cube = cube[:, :, np.newaxis]
        return cube

    if cube.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D data in .b2nd file, got shape {cube.shape}."
        )

    # Heuristic: if first dimension is smallest, interpret as (bands, H, W)
    if cube.shape[0] <= cube.shape[1] and cube.shape[0] <= cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))

    return cube.astype(np.float32)

