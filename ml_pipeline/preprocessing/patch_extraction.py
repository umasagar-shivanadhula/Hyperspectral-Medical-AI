"""
Patch Extraction — Unified HSI Medical Framework
=================================================

Grid-based (stride) patch extraction from 3-D hyperspectral cubes.

Key rules (enforced by this module):
  • Patches are NEVER flattened — shape is always (patch_size, patch_size, B)
  • Extraction is deterministic (grid-based), NOT random
  • Radiometric normalisation must be applied BEFORE calling extract_patches
  • Spectral band selection is applied after normalisation if requested

Public API
----------
extract_patches(cube, patch_size, stride)
    Yield (row, col, patch) triples — generator, memory-friendly.

extract_patches_list(cube, patch_size, stride, max_patches)
    Return list of (patch_size, patch_size, B) arrays (capped at max_patches).

select_informative_bands(cube, n_bands)
    Return cube with top-N variance bands retained.
"""

import logging
import numpy as np
from typing import Generator, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Band selection
# ──────────────────────────────────────────────────────────────────────────────

def select_informative_bands(cube: np.ndarray, n_bands: int = 0) -> np.ndarray:
    """
    Select the top-N most informative spectral bands by spatial variance.

    Parameters
    ----------
    cube    : (H, W, B) float32 array
    n_bands : number of bands to keep; 0 = keep all (no-op)

    Returns
    -------
    (H, W, n_bands) cube  — original order preserved (bands sorted by index)
    """
    if cube.ndim != 3:
        raise ValueError(f"select_informative_bands expects (H,W,B), got {cube.shape}")

    H, W, B = cube.shape

    if n_bands <= 0 or n_bands >= B:
        return cube   # no-op

    # Per-band spatial variance
    flat     = cube.reshape(-1, B)        # (H*W, B)
    variance = flat.var(axis=0)            # (B,)

    top_idx  = np.argsort(variance)[::-1][:n_bands]   # top-N indices
    top_idx  = np.sort(top_idx)                         # restore spectral order

    selected = cube[:, :, top_idx]
    logger.info(
        "Band selection: kept %d / %d bands  (top by variance, indices=%s)",
        n_bands, B, top_idx.tolist()
    )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Grid-based patch extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_patches(
    cube:       np.ndarray,
    patch_size: int = 32,
    stride:     int = 16,
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Deterministic, grid-based patch extraction with full-coverage border padding.

    Yields
    ------
    (row_start, col_start, patch)  where patch.shape == (patch_size, patch_size, B)

    Notes
    -----
    • Patches are 3-D arrays — they are NEVER flattened here.
    • Radiometric correction must be applied to *cube* before calling this.
    • The cube is reflect-padded so that the patch grid covers every pixel,
      including pixels in the rightmost columns and bottom rows that would
      otherwise be skipped when (H - patch_size) % stride != 0.
    • Padding is also applied when the cube is smaller than patch_size.
    """
    if cube.ndim != 3:
        raise ValueError(
            f"extract_patches expects a 3-D (H,W,B) cube, got shape {cube.shape}"
        )

    H, W, B = cube.shape

    # ── Border padding — ensures every pixel is covered ──────────────────────
    #
    # For a cube of size H×W with patch_size P and stride S, the last patch
    # starts at position floor((H-P)/S)*S.  Any rows/cols beyond that point
    # are silently dropped without padding.  We pad so the final patch reaches
    # the true edge.
    #
    # Two cases are handled together:
    #   (a) cube smaller than one patch  → pad to at least patch_size
    #   (b) cube larger but remainder doesn't align with stride
    #       → pad the tail so the last patch lands exactly on the border
    #
    # The same reflect padding is used in both cases (no data fabrication).

    def _needed_pad(size: int) -> int:
        """Extra pixels needed so grid covers the full dimension."""
        if size < patch_size:
            return patch_size - size       # case (a)
        remainder = (size - patch_size) % stride
        return (stride - remainder) % stride   # case (b); 0 if already aligned

    pad_h = _needed_pad(H)
    pad_w = _needed_pad(W)

    if pad_h > 0 or pad_w > 0:
        logger.debug(
            "Border padding: cube (%d×%d) pad_h=%d pad_w=%d  "
            "(patch_size=%d, stride=%d)",
            H, W, pad_h, pad_w, patch_size, stride,
        )
        cube = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        H, W, _ = cube.shape

    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    total  = n_rows * n_cols

    logger.info(
        "Patch extraction: cube=%s  patch_size=%d  stride=%d  → %d patches",
        (H, W, B), patch_size, stride, total
    )

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = cube[r : r + patch_size, c : c + patch_size, :]   # (P,P,B) — NO flatten
            if patch.shape != (patch_size, patch_size, B):
                raise RuntimeError(
                    f"Invalid patch extraction: expected shape "
                    f"({patch_size}, {patch_size}, {B}), got {patch.shape}. "
                    "This indicates a bug in the border-padding logic."
                )
            yield (r, c, patch)


def extract_patches_list(
    cube:       np.ndarray,
    patch_size: int = 32,
    stride:     int = 16,
    max_patches: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Convenience wrapper: returns list of (patch_size, patch_size, B) arrays.
    Optionally capped at max_patches (first N in raster order).
    """
    patches = []
    for (_, _, patch) in extract_patches(cube, patch_size, stride):
        patches.append(patch)
        if max_patches is not None and len(patches) >= max_patches:
            break

    if not patches:
        logger.warning("No patches extracted; returning full cube as single patch.")
        patches = [cube]

    logger.info("extract_patches_list → %d patches of shape %s", len(patches), patches[0].shape)
    return patches
