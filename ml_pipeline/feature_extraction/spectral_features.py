"""
Spectral Feature Extraction for HSI Medical Data.

Computes spectral statistics and indices from hyperspectral cubes
or extracted patches.
"""

import numpy as np
from typing import Tuple


def extract_spectral_features(cube: np.ndarray) -> np.ndarray:
    """
    Extract full spectral feature vector from an HSI patch or cube.

    Features:
      - Mean reflectance per band            (B,)
      - Variance per band                    (B,)
      - Standard deviation per band          (B,)
      - Skewness per band                    (B,)
      - Kurtosis per band                    (B,)
      - Spectral NDI between adjacent bands  (B-1,)
      - Global spectral slope                (1,)
      - Spectral peak position               (1,)

    Args:
        cube: (H, W, B) cube OR (N, B) flattened spectral data

    Returns:
        1D feature vector
    """

    # --------------------------------------------------
    # Handle cube OR flattened patches
    # --------------------------------------------------

    if cube.ndim == 3:
        H, W, B = cube.shape
        flat = cube.reshape(-1, B)

    elif cube.ndim == 2:
        flat = cube
        B = flat.shape[1]

    else:
        raise ValueError(f"Unsupported cube shape: {cube.shape}")

    # --------------------------------------------------
    # Spectral statistics
    # --------------------------------------------------

    mean = flat.mean(axis=0)
    var = flat.var(axis=0)
    std = flat.std(axis=0)

    skew = _skewness(flat)
    kurt = _kurtosis(flat)

    # --------------------------------------------------
    # Normalized Difference Index between adjacent bands
    # --------------------------------------------------

    ndi = []

    for i in range(B - 1):
        denom = mean[i] + mean[i + 1] + 1e-8
        ndi.append((mean[i] - mean[i + 1]) / denom)

    ndi = np.array(ndi)

    # --------------------------------------------------
    # Global spectral slope
    # --------------------------------------------------

    band_idx = np.arange(B, dtype=float)

    if B > 1:
        slope = np.polyfit(band_idx, mean, 1)[0]
    else:
        slope = 0.0

    peak = float(np.argmax(mean))

    return np.concatenate([
        mean,
        var,
        std,
        skew,
        kurt,
        ndi,
        np.array([slope, peak])
    ]).astype(np.float32)


def _skewness(flat: np.ndarray) -> np.ndarray:
    """Compute per-band skewness across pixels."""

    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    return ((flat - mean) ** 3).mean(axis=0) / (std ** 3)


def _kurtosis(flat: np.ndarray) -> np.ndarray:
    """Compute per-band kurtosis across pixels."""

    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    return ((flat - mean) ** 4).mean(axis=0) / (std ** 4) - 3.0


def compute_spectral_indices(cube: np.ndarray) -> dict:
    """
    Compute medically relevant spectral indices from reflectance cube.

    Returns dict with:
      - oxy_index
      - dehb_index
      - ndvi_proxy
    """

    if cube.ndim != 3:
        raise ValueError("Spectral indices require full cube (H,W,B)")

    H, W, B = cube.shape

    indices = {}

    # Oxygenation index
    if B >= 16:

        high_bands = cube[:, :, 12:16].mean(axis=2)
        low_bands = cube[:, :, :4].mean(axis=2)

        denom = high_bands + low_bands + 1e-8

        indices["oxy_index"] = (high_bands - low_bands) / denom
        indices["dehb_index"] = 1.0 - indices["oxy_index"]

    else:

        mid = B // 2

        upper = cube[:, :, mid:].mean(axis=2)
        lower = cube[:, :, :mid].mean(axis=2)

        denom = upper + lower + 1e-8

        indices["oxy_index"] = (upper - lower) / denom
        indices["dehb_index"] = 1.0 - indices["oxy_index"]

    # NDVI proxy

    b1 = max(0, B // 3 - 1)
    b2 = min(B - 1, 2 * B // 3)

    r = cube[:, :, b1]
    nir = cube[:, :, b2]

    denom = nir + r + 1e-8

    indices["ndvi_proxy"] = (nir - r) / denom

    return indices


def band_selection_variance(cube: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """
    Select top spectral bands based on spatial variance.
    """

    H, W, B = cube.shape

    flat = cube.reshape(-1, B)

    var = flat.var(axis=0)

    top_idx = np.argsort(var)[::-1][:n_bands]

    top_idx = np.sort(top_idx)

    return cube[:, :, top_idx]


def compute_spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spectral Angle Mapper (SAM).
    """

    dot = np.dot(a, b)

    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10

    return float(np.arccos(np.clip(dot / norm, -1.0, 1.0)))


def pca_reduce(features: np.ndarray, n_components: int = 20) -> Tuple[np.ndarray, object]:
    """
    Apply PCA for dimensionality reduction.
    """

    try:

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        features_scaled = scaler.fit_transform(features)

        n_comp = min(n_components, features.shape[1], features.shape[0])

        pca = PCA(n_components=n_comp, svd_solver="full")

        reduced = pca.fit_transform(features_scaled)

        return reduced, (scaler, pca)

    except ImportError:

        return features, None