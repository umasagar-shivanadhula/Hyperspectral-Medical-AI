"""
ENVI Hyperspectral Image Loader using Spectral Python.

Supports .raw and .hdr file pairs. Converts hyperspectral images to numpy arrays
with shape: height × width × spectral_bands.
"""
import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def load_hyperspectral_image(
    path: Union[str, Path],
    hdr_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Load ENVI hyperspectral image using spectral.open_image().

    Expects either:
      - path: path to .hdr file (raw file must be same basename, no extension or .raw)
      - path: path to .raw file with hdr_path pointing to .hdr file

    Returns:
        numpy array of shape (height, width, spectral_bands), dtype float32.
    """
    path = Path(path)
    if hdr_path is not None:
        hdr_path = Path(hdr_path)
    else:
        if path.suffix.lower() == ".hdr":
            hdr_path = path
        elif path.suffix.lower() in (".raw", "") or path.name == "raw":
            # Look for sibling .hdr
            hdr_path = path.parent / (path.stem + ".hdr")
            if not hdr_path.exists():
                hdr_path = path.parent / (path.name + ".hdr")
            if not hdr_path.exists():
                raise FileNotFoundError(f"No .hdr file found for raw file: {path}")
        else:
            hdr_path = path.with_suffix(".hdr")
            if not hdr_path.exists():
                raise FileNotFoundError(f"No .hdr file found: {hdr_path}")

    if not hdr_path.exists():
        raise FileNotFoundError(f"Header file not found: {hdr_path}")

    try:
        import spectral
    except ImportError:
        raise ImportError("Spectral Python is required: pip install spectral")

    img = spectral.open_image(str(hdr_path))
    if img is None:
        raise IOError(f"Failed to open hyperspectral image: {hdr_path}")

    # Load full cube: spectral returns (rows, cols, bands) for BIP; (bands, rows, cols) for BSQ
    cube = np.asarray(img.load())
    if cube is None or cube.size == 0:
        raise IOError(f"Failed to load cube from: {hdr_path}")

    if cube.ndim == 2:
        cube = cube[:, :, np.newaxis]
    elif cube.ndim == 3:
        # If first dim is smallest, likely (bands, rows, cols) -> (rows, cols, bands)
        if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
            cube = np.transpose(cube, (1, 2, 0))

    return np.asarray(cube, dtype=np.float32)


def load_hyperspectral_from_bytes(
    data: bytes,
    ext: str,
    filename: str,
    hdr_bytes: Optional[bytes] = None,
) -> np.ndarray:
    """
    Load hyperspectral cube from uploaded bytes (API use).

    Supports:
      - .npy: numpy array (H, W, B)
      - .npz: numpy archive, first array key
      - .h5: h5py file, first key
      - .hdr + raw: when hdr_bytes is provided, or when ext is .hdr and data is .hdr content
        and raw is in a second upload (use two-file upload for ENVI)

    For ENVI: pass data as raw file bytes and hdr_bytes as .hdr file content when
    uploading both files. Or pass a single .npy/.npz for pre-converted data.
    """
    ext = ext.lower()
    if ext == ".npy":
        return np.load(io.BytesIO(data)).astype(np.float32)
    if ext == ".npz":
        npz = np.load(io.BytesIO(data))
        key = list(npz.files)[0]
        return npz[key].astype(np.float32)
    if ext == ".h5":
        try:
            import h5py
            with h5py.File(io.BytesIO(data), "r") as f:
                key = list(f.keys())[0]
                return np.asarray(f[key], dtype=np.float32)
        except ImportError:
            raise RuntimeError("h5py not installed. pip install h5py")

    if ext in (".hdr", ".raw", ".bin") or ext == "":
        # ENVI: need both .hdr and raw data in temp dir
        if hdr_bytes is None and ext == ".hdr":
            # Single file upload of .hdr only — cannot load without raw
            raise ValueError(
                "ENVI .hdr upload requires raw data. Upload both .hdr and .raw (or raw binary) files."
            )
        with tempfile.TemporaryDirectory(prefix="hsi_") as tmpdir:
            tmp = Path(tmpdir)
            if ext == ".hdr" and hdr_bytes is not None:
                hdr_path = tmp / "image.hdr"
                raw_path = tmp / "image"
                hdr_path.write_bytes(hdr_bytes)
                raw_path.write_bytes(data)
            elif ext in (".raw", ".bin", "") and hdr_bytes is not None:
                base = Path(filename).stem.replace(".raw", "").replace(".bin", "")
                hdr_path = tmp / f"{base}.hdr"
                raw_path = tmp / base
                hdr_path.write_bytes(hdr_bytes)
                raw_path.write_bytes(data)
            elif ext == ".hdr":
                # data is .hdr content, we don't have raw
                raise ValueError("Upload raw binary file together with .hdr for ENVI format.")
            else:
                # Try writing as single raw and look for .hdr in same upload — we need hdr_bytes
                base = Path(filename).stem
                if not base or base == filename:
                    base = "image"
                hdr_path = tmp / f"{base}.hdr"
                raw_path = tmp / base
                raw_path.write_bytes(data)
                if hdr_bytes:
                    hdr_path.write_bytes(hdr_bytes)
                else:
                    raise ValueError("ENVI format requires both .hdr and raw data files.")
            return load_hyperspectral_image(hdr_path)

    raise ValueError(f"Unsupported file type: {ext}. Use .npy, .npz, .h5, or ENVI .hdr+.raw.")


def get_cube_shape(cube: np.ndarray) -> Tuple[int, int, int]:
    """Return (height, width, bands) from cube."""
    if cube.ndim == 2:
        return cube.shape[0], cube.shape[1], 1
    return cube.shape[0], cube.shape[1], cube.shape[2]
