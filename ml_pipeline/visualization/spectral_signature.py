
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_spectral_signatures(tumor_pixels, normal_pixels, wavelengths, output_path):

    tumor_mean = np.mean(tumor_pixels, axis=0)
    normal_mean = np.mean(normal_pixels, axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,5))

    plt.plot(wavelengths, tumor_mean, label="Tumor", linewidth=2)
    plt.plot(wavelengths, normal_mean, label="Normal", linewidth=2)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")

    plt.title("Spectral Signature Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
