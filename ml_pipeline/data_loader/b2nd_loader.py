"""
Loader utilities for Blosc2 B2ND (.b2nd) hyperspectral files.

This module provides a thin wrapper around python-blosc2 so that .b2nd cubes
can be loaded as NumPy arrays with shape (H, W, B) and dtype float32, matching
the expectations of the rest of the pipeline.
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

from pathlib import Path
from typing import Union

def load_b2nd_cube(path: Union[str, Path]):
    """
    Returns a memory-mapped Blosc2 array.
    Does NOT load full cube into RAM.
    """
    path = Path(path)
    blosc2 = _ensure_blosc2()

    nda = blosc2.open(str(path), mode="r")

    # DO NOT convert to numpy here
    return nda