"""
Spatial Feature Extraction for HSI Medical Data.

Computes GLCM-based texture features and spatial statistics
from hyperspectral image patches.
"""
import numpy as np
from typing import List


def extract_spatial_features(cube: np.ndarray, n_bands_use: int = 4) -> np.ndarray:
    """
    Extract spatial texture features from a hyperspectral patch.

    For each of the first n_bands_use bands, computes:
      - GLCM Haralick features (contrast, energy, entropy, homogeneity, correlation)
      - Local Binary Pattern histogram approximation
      - Spatial statistics (mean, std, min, max of gradients)

    Args:
        cube:        (H, W, B) float32 patch
        n_bands_use: number of bands to compute spatial features for

    Returns:
        1D feature vector
    """
    H, W, B = cube.shape
    n_bands_use = min(n_bands_use, B)
    all_features = []

    for b in range(n_bands_use):
        band = cube[:, :, b]
        glcm_feats = compute_glcm_features(band)
        gradient_feats = compute_gradient_features(band)
        lbp_feats = compute_lbp_features(band)
        all_features.extend(glcm_feats)
        all_features.extend(gradient_feats)
        all_features.extend(lbp_feats)

    return np.array(all_features, dtype=np.float32)


def compute_glcm_features(band: np.ndarray, levels: int = 8) -> List[float]:
    """
    Compute Haralick GLCM features for a 2D band image.
    Uses scikit-image when available (graycomatrix, graycoprops), else manual implementation.

    Computes GLCM for 4 directions: 0°, 45°, 90°, 135°.
    Returns mean Haralick features: contrast, energy, entropy, homogeneity, correlation, dissimilarity
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
        mn, mx = band.min(), band.max()
        if mx - mn > 1e-8:
            q = np.clip(((band - mn) / (mx - mn) * (levels - 1)).astype(np.uint8), 0, levels - 1)
        else:
            q = np.zeros_like(band, dtype=np.uint8)
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        energy = float(np.mean(graycoprops(glcm, "energy")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        # entropy and correlation not in graycoprops; compute from GLCM
        p = np.squeeze(glcm)
        if p.ndim == 3:
            p = np.mean(p, axis=2)
        p = p / (p.sum() + 1e-10)
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)
        sig_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * p) + 1e-10)
        sig_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * p) + 1e-10)
        correlation = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * p) / (sig_i * sig_j))
        dissimilarity = float(np.sum(np.abs(i_idx - j_idx) * p))
        return [contrast, energy, entropy, homogeneity, dissimilarity, correlation]
    except ImportError:
        pass
    # Manual fallback
    mn, mx = band.min(), band.max()
    if mx - mn > 1e-8:
        q = np.clip(((band - mn) / (mx - mn) * (levels - 1)).astype(int), 0, levels - 1)
    else:
        q = np.zeros_like(band, dtype=int)

    H, W = q.shape
    # Direction offsets: (dy, dx)
    directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    features_all_dirs = []

    for dy, dx in directions:
        glcm = np.zeros((levels, levels), dtype=np.float64)
        # Extract valid pairs
        r0 = max(0, -dy); r1 = H - max(0, dy)
        c0 = max(0, -dx); c1 = W - max(0, dx)
        rows_i = q[r0:r1, c0:c1]
        rows_j = q[r0 + dy:r1 + dy, c0 + dx:c1 + dx]

        for i in range(levels):
            for j in range(levels):
                glcm[i, j] = ((rows_i == i) & (rows_j == j)).sum()

        # Symmetrize and normalize
        glcm = glcm + glcm.T
        total = glcm.sum() + 1e-10
        p = glcm / total

        # Compute Haralick features
        i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")

        contrast = float(np.sum((i_idx - j_idx) ** 2 * p))
        energy = float(np.sum(p ** 2))
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        homogeneity = float(np.sum(p / (1.0 + np.abs(i_idx - j_idx))))
        dissimilarity = float(np.sum(np.abs(i_idx - j_idx) * p))

        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)
        sig_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * p) + 1e-10)
        sig_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * p) + 1e-10)
        correlation = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * p) / (sig_i * sig_j))

        features_all_dirs.append([contrast, energy, entropy, homogeneity, dissimilarity, correlation])

    # Average over directions
    avg = np.mean(features_all_dirs, axis=0)
    return avg.tolist()


def compute_gradient_features(band: np.ndarray) -> List[float]:
    """
    Compute gradient magnitude statistics for a 2D band.

    Returns: mean, std, max, entropy of gradient magnitudes
    """
    # Sobel-like gradients
    if band.shape[0] < 2 or band.shape[1] < 2:
        return [0.0, 0.0, 0.0, 0.0]

    gy = np.diff(band, axis=0)[:, :-1] if band.shape[1] > 1 else np.diff(band, axis=0)
    gx = np.diff(band, axis=1)[:-1, :] if band.shape[0] > 1 else np.diff(band, axis=1)

    # Align shapes
    min_h = min(gy.shape[0], gx.shape[0])
    min_w = min(gy.shape[1], gx.shape[1])
    mag = np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2).ravel()

    if mag.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(mag.mean()),
        float(mag.std()),
        float(mag.max()),
        float(-np.sum((mag / (mag.sum() + 1e-10)) * np.log(mag / (mag.sum() + 1e-10) + 1e-12)))
    ]


def compute_lbp_features(band: np.ndarray, n_bins: int = 8) -> List[float]:
    """
    Compute an approximate Local Binary Pattern histogram.

    Uses 3×3 neighborhoods with 8 neighbors.
    Returns normalized histogram of LBP codes.
    """
    H, W = band.shape
    if H < 3 or W < 3:
        return [0.0] * n_bins

    lbp = np.zeros((H - 2, W - 2), dtype=np.uint8)
    # 8 neighbors in circular order for 3x3
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]

    center = band[1:-1, 1:-1]
    for bit, (dr, dc) in enumerate(offsets):
        neighbor = band[1 + dr:H - 1 + dr, 1 + dc:W - 1 + dc]
        lbp |= (neighbor >= center).astype(np.uint8) << bit

    # Histogram with n_bins bins over [0, 256)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 256))
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-10
    return hist.tolist()


def extract_joint_features(cube: np.ndarray) -> np.ndarray:
    """
    Combine spectral and spatial features for joint classification.

    Args:
        cube: (H, W, B) patch

    Returns:
        Concatenated feature vector
    """
    from ml_pipeline.feature_extraction.spectral_features import extract_spectral_features
    spectral = extract_spectral_features(cube)
    spatial = extract_spatial_features(cube)
    return np.concatenate([spectral, spatial]).astype(np.float32)
