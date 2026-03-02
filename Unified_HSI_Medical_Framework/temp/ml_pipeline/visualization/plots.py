"""
Visualization Module for HSI Medical Analysis.

Generates spectral signature plots, perfusion heatmaps,
tumor probability maps, and classifier comparison charts.
"""
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[2] / "outputs"


def save_spectral_signature_plot(
    bands: list,
    tissue_spectra: np.ndarray,
    healthy_spectra: np.ndarray,
    title: str = "Spectral Signature",
    output_path: str = None
):
    """
    Plot mean spectral signatures for tissue vs healthy reference.

    Args:
        bands:          list of band indices
        tissue_spectra: (B,) mean reflectance for tissue sample
        healthy_spectra:(B,) mean reflectance for healthy reference
        title:          plot title
        output_path:    save path (PNG)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.style as style
        style.use("dark_background")

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#080f1e")
        ax.set_facecolor("#080f1e")

        ax.plot(bands, tissue_spectra, color="#ff4466", linewidth=2,
                marker="o", markersize=4, label="Tissue Sample")
        ax.fill_between(bands, tissue_spectra, alpha=0.15, color="#ff4466")

        ax.plot(bands, healthy_spectra, color="#00e5ff", linewidth=2,
                marker="s", markersize=4, label="Healthy Reference")
        ax.fill_between(bands, healthy_spectra, alpha=0.08, color="#00e5ff")

        ax.set_xlabel("Spectral Band", color="#8090b0", fontsize=11)
        ax.set_ylabel("Reflectance (%)", color="#8090b0", fontsize=11)
        ax.set_title(title, color="#c8dcff", fontsize=13, pad=15)
        ax.tick_params(colors="#6080aa")
        ax.legend(facecolor="#0d1830", edgecolor="#0060aa", labelcolor="#c8dcff")
        ax.grid(color="#0a1830", linewidth=0.5)
        ax.spines[:].set_color("#0a2040")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None


def save_heatmap(
    data: np.ndarray,
    colormap: str = "RdYlGn",
    title: str = "Spatial Heatmap",
    output_path: str = None
):
    """
    Save a spatial heatmap as PNG.

    Args:
        data:       (H, W) float array [0, 1]
        colormap:   matplotlib colormap name
        title:      plot title
        output_path: save path
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#080f1e")
        ax.set_facecolor("#080f1e")

        im = ax.imshow(data, cmap=colormap, vmin=0, vmax=1, interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="#c8dcff", fontsize=12)
        ax.axis("off")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig
    except ImportError:
        return None


def save_probability_chart(
    class_names: list,
    probabilities: list,
    title: str = "Fusion Probabilities",
    output_path: str = None
):
    """
    Save a horizontal bar chart of class probabilities.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#080f1e")
        ax.set_facecolor("#080f1e")

        colors = ["#00ff88", "#ff8c00", "#ff2244", "#0096ff"][:len(class_names)]
        bars = ax.barh(class_names, probabilities, color=colors, alpha=0.8, height=0.5)

        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1f}%", va="center", color="#c8dcff", fontsize=11)

        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)", color="#8090b0")
        ax.set_title(title, color="#c8dcff", fontsize=12)
        ax.tick_params(colors="#6080aa")
        ax.spines[:].set_color("#0a2040")
        ax.set_facecolor("#080f1e")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig
    except ImportError:
        return None


def generate_rgb_composite(cube: np.ndarray) -> np.ndarray:
    """
    Create an RGB false-color composite from hyperspectral cube.
    Maps 3 bands to R, G, B channels.

    Args:
        cube: (H, W, B)

    Returns:
        (H, W, 3) uint8 RGB image
    """
    H, W, B = cube.shape
    r_idx = min(B - 1, 2 * B // 3)
    g_idx = min(B - 1, B // 2)
    b_idx = min(B - 1, B // 4)

    def normalize(band):
        mn, mx = band.min(), band.max()
        if mx - mn < 1e-8:
            return np.zeros_like(band)
        return ((band - mn) / (mx - mn) * 255).astype(np.uint8)

    r = normalize(cube[:, :, r_idx])
    g = normalize(cube[:, :, g_idx])
    b_ch = normalize(cube[:, :, b_idx])

    return np.stack([r, g, b_ch], axis=2)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    output_path: str = None
):
    """Save confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#080f1e")
        ax.set_facecolor("#080f1e")

        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right", color="#c8dcff", fontsize=9)
        ax.set_yticklabels(class_names, color="#c8dcff", fontsize=9)

        thresh = cm.max() / 2
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] < thresh else "black", fontsize=11)

        ax.set_title(title, color="#c8dcff", fontsize=12)
        ax.set_xlabel("Predicted", color="#8090b0")
        ax.set_ylabel("Actual", color="#8090b0")
        ax.tick_params(colors="#6080aa")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig
    except ImportError:
        return None
